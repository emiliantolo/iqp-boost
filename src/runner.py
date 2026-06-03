"""Common runner for ensemble boosting experiments."""

import iqpopt as iqp
from src.sigma_heuristics import compute_sigma
from iqpopt.gen_qml.iqp_methods import mmd_loss_iqp
from src.ensemble import BoostedEnsemble
try:
    from src.circuit_artifacts import save_circuit_artifact
except ModuleNotFoundError:
    save_circuit_artifact = None
from src.reporting import (
    get_plot_config, OutputManager, report_baseline, report_rejection,
    report_step, report_gradient_snr, report_config, report_circuit, report_kernel,
    report_loss_components, report_acceptance, save_circuit_plot
)
from src.core import (
    setup_iqp_circuit, get_params_init, compute_lambda_schedule
)
from src.dataset_catalog import DatasetBundle
from src.evaluation import EvaluationPolicy
from src.final_evaluation import FinalEvaluationContext, run_final_evaluation
from src.dual_mmd_loss import gradient_snr, dual_mmd_loss, EnsembleTerms
import jax
import numpy as np
import gc


def compute_dual_components_from_traces(traces: dict, n_samples: int, n_data: int) -> dict:
    """Reconstruct dual loss components from a trace pass.

    Uses the same unbiased estimators as dual_loss_estimate_iqp, but avoids a
    second circuit evaluation when traces were already requested for line search.
    """
    data_vals = []
    ensemble_vals = []
    total_vals = []

    for tr_data, tr_ens, tr_iqp, corr_ens, corr_iqp in zip(
        traces['trs_data'],
        traces['trs_ens'],
        traces['trs_iqp'],
        traces['trs_corr_ens'],
        traces['trs_corr_iqp'],
    ):
        tr_data = np.asarray(tr_data)
        tr_ens = np.asarray(tr_ens)
        tr_iqp = np.asarray(tr_iqp)
        corr_ens = np.asarray(corr_ens)
        corr_iqp = np.asarray(corr_iqp)

        term_p_p = np.mean((tr_iqp * tr_iqp - corr_iqp) * n_samples / (n_samples - 1))
        cross_data = np.mean(tr_iqp * tr_data)
        cross_ens = np.mean(tr_iqp * tr_ens)
        term_data_data = np.mean((tr_data * tr_data * n_data - 1) / (n_data - 1))
        term_ens_ens = np.mean((tr_ens * tr_ens - corr_ens) * n_samples / (n_samples - 1))

        data_val = term_p_p - 2.0 * cross_data + term_data_data
        ensemble_val = term_p_p - 2.0 * cross_ens + term_ens_ens

        data_vals.append(data_val)
        ensemble_vals.append(ensemble_val)
        total_vals.append(data_val - ensemble_val)

    return {
        'total': float(np.mean(total_vals)),
        'data': float(np.mean(data_vals)),
        'ensemble': float(np.mean(ensemble_vals)),
    }


def check_acceptance(current_metric: float, previous_metric: float, ensemble: BoostedEnsemble,
                     snapshot: dict, step: int, keep_all: bool = False,
                     stop_on_reject: bool = False) -> tuple[bool, bool]:
    """Check if new model should be accepted based on metric improvement."""
    delta_mmd = previous_metric - current_metric
    accepted = delta_mmd > 0

    if keep_all:
        report_acceptance(True, delta_mmd, diagnostic_mode=True)
        return True, False

    if not accepted:
        report_rejection(step)
        ensemble.restore_state(snapshot)
        return False, stop_on_reject

    return True, False


def train_standalone_model(circuit: iqp.IqpSimulator, x_train: np.ndarray, key: jax.Array,
                           sigma: float | list, n_ops: int, n_samples: int, config: dict,
                           epochs: int, monitor_interval: int | None, turbo_opt: int | None,
                           wires: list = None) -> tuple:
    """Train a single standalone IQP model using iqpopt's mmd_loss_iqp directly."""
    key, init_key = jax.random.split(key, 2)
    params_init = get_params_init(config.get('init_baseline', 'random'), circuit, x_train, init_key)

    loss_kwargs = {
        "params": params_init, "iqp_circuit": circuit, "ground_truth": x_train,
        "sigma": sigma, "n_ops": n_ops, "n_samples": n_samples,
        "wires": wires,
        "max_batch_ops": config.get('max_batch_ops', None),
        "max_batch_samples": config.get('max_batch_samples', None),
    }

    trainer = iqp.Trainer("Adam", mmd_loss_iqp, stepsize=config['learning_rate'])
    trainer.train(n_iters=epochs, loss_kwargs=loss_kwargs,
                  monitor_interval=monitor_interval, turbo=turbo_opt)

    return key, trainer, trainer.final_params


def train_ensemble_model_0(ensemble: BoostedEnsemble, x_train: np.ndarray, key: jax.Array,
                           config: dict, monitor_interval: int | None, turbo_opt: int | None) -> tuple:
    """Train the first model (Model 0) of the ensemble."""
    key, init_key = jax.random.split(key, 2)
    params_init = get_params_init(config.get('init_baseline', 'random'), ensemble.iqp_circuit, x_train, init_key)

    caching_level = config.get('caching_level', 'none')

    stochastic_ops = (caching_level == 'none')

    # Seed the initial cache of operators BEFORE Model 0 trains
    key, ops_key = jax.random.split(key)
    ensemble.terms.sample_ops(ensemble.iqp_circuit, ensemble.sigma, ensemble.n_ops, ops_key, wires=ensemble.wires)

    loss_kwargs = {
        "params": params_init, "iqp_circuit": ensemble.iqp_circuit, "weights": [],
        "ground_truth": x_train, "ensemble_terms": ensemble.terms,
        "sigma": ensemble.sigma, "n_ops": ensemble.n_ops, "n_samples": ensemble.n_samples,
        "lambda_dual": ensemble.lambda_dual, "key": key,
        "stochastic_ops": stochastic_ops,
        "ensemble_models": [],
        "wires": ensemble.wires,
        "max_batch_ops": config.get('max_batch_ops', None),
    }

    trainer = iqp.Trainer("Adam", dual_mmd_loss, stepsize=config['learning_rate'])
    epochs = config.get('epochs_per_step', 250)
    trainer.train(n_iters=epochs, loss_kwargs=loss_kwargs,
                  monitor_interval=monitor_interval, turbo=turbo_opt)

    key, subkey = jax.random.split(key, 2)
    schedule = config.get('lambda_schedule') or {}
    gamma = float(schedule.get('gamma', 2.0))
    tau = float(schedule.get('tau', 2.0))
    alpha = ensemble.add_model(trainer.final_params, subkey, gamma=gamma, tau=tau)

    ensemble.training_losses.append({
        'total': np.array(trainer.losses),
        'data_final': None,
        'ensemble_final': None
    })

    return key, trainer, alpha


def train_boosting_step(ensemble: BoostedEnsemble, x_train: np.ndarray, key: jax.Array,
                        config: dict, monitor_interval: int | None, turbo_opt: int | None,
                        snapshot: dict, step: int = 0,
                        validity_fn: callable = None,
                        coverage_fn: callable = None) -> tuple:
    """Train a new model in the context of the existing ensemble."""
    key, step_key = jax.random.split(key)
    step_key, init_key = jax.random.split(step_key)

    params_init = get_params_init(config['init_later'], ensemble.iqp_circuit, x_train, init_key)

    # 1. Determine Caching Level
    caching_level = config.get('caching_level', 'none')

    # 2. Apply Caching Strategy
    if caching_level == 'step' and ensemble.models:
        # Refresh the fixed cache once per boosting step
        ensemble.refresh_terms(step_key)
        
    stochastic_ops = (caching_level == 'none')

    loss_kwargs = {
        "params": params_init, "iqp_circuit": ensemble.iqp_circuit, "weights": ensemble.weights,
        "ground_truth": x_train, "ensemble_terms": ensemble.terms,
        "sigma": ensemble.sigma, "n_ops": ensemble.n_ops, "n_samples": ensemble.n_samples,
        "lambda_dual": ensemble.lambda_dual, "key": step_key,
        "stochastic_ops": stochastic_ops,
        "ensemble_models": ensemble.models,
        "wires": ensemble.wires,
        "max_batch_ops": config.get('max_batch_ops', None),
        "max_batch_samples": config.get('max_batch_samples', None),
    }

    # Use monitor_interval=turbo to fetch historical params for plotting components
    monitor_interval = config.get('monitor_interval', turbo_opt)

    trainer = iqp.Trainer("Adam", dual_mmd_loss, stepsize=config['learning_rate'])
    trainer.train(n_iters=config['epochs_per_step'], loss_kwargs=loss_kwargs,
                  monitor_interval=monitor_interval, turbo=turbo_opt)

    # Reconstruct data/ensemble dual loss components using params_hist without needing a custom Trainer
    data_hist = []
    ens_hist = []
    hist_epochs = []
    if hasattr(trainer, 'params_hist') and trainer.params_hist is not None and len(trainer.params_hist) > 0:
        actual_params = []
        for ep, p in enumerate(trainer.params_hist):
            # iqpopt turbo mode can fill non-monitored steps with exact zeros
            if not np.all(p == 0):
                actual_params.append(p)
                hist_epochs.append(ep)
                
        if actual_params:
            eval_keys = jax.random.split(step_key, len(actual_params))
            for p, k in zip(actual_params, eval_keys):
                comp = dual_mmd_loss(
                    p, ensemble.iqp_circuit, x_train, ensemble.terms,
                    ensemble.weights, ensemble.sigma, ensemble.n_ops, ensemble.n_samples, k,
                    lambda_dual=ensemble.lambda_dual, return_components=True,
                    stochastic_ops=stochastic_ops, ensemble_models=ensemble.models,
                    wires=ensemble.wires,
                    max_batch_ops=ensemble.max_batch_ops,
                    max_batch_samples=ensemble.max_batch_samples
                )
                data_hist.append(float(comp['data']))
                ens_hist.append(float(comp['ensemble']))

    # Final eval with traces for line search
    traces = dual_mmd_loss(
        trainer.final_params, ensemble.iqp_circuit, x_train, ensemble.terms,
        ensemble.weights, ensemble.sigma, ensemble.n_ops, ensemble.n_samples, step_key,
        lambda_dual=ensemble.lambda_dual, return_traces=True, wires=ensemble.wires,
        stochastic_ops=stochastic_ops, ensemble_models=ensemble.models,
        max_batch_ops=ensemble.max_batch_ops, max_batch_samples=ensemble.max_batch_samples
    )

    final_comp = compute_dual_components_from_traces(
        traces,
        n_samples=ensemble.n_samples,
        n_data=len(x_train),
    )

    # Sample-based strategies (TVD / validity / coverage) need samples drawn from
    # the OLD ensemble and from the new model *before* add_model() mutates the ensemble.
    # Use a separate step-derived RNG so weight search sampling doesn't consume
    # the evaluation RNG (which would cause metric fluctuations between steps).
    weight_strategy = config.get('weight_strategy', 'greedy')
    alpha_n_grid = int(config.get('alpha_n_grid', 11))
    alpha_objective_weight = float(config.get('alpha_objective_weight', 0.5))
    _sample_based = {'tvd_line_search', 'validity_line_search', 'coverage_line_search', 'sum_line_search'}
    samples_old_for_search = None
    samples_new_for_search = None
    if weight_strategy in _sample_based and ensemble.models:
        search_rng = np.random.default_rng(config.get('rng_seed', 0) + step * 7919)
        n_shots_search = int(config.get('shots', 1000))
        samples_old_for_search = ensemble.sample(n_shots_search, search_rng)
        # Backward-compatible sampling: older simulator versions don't accept
        # a `wires` keyword, so sample full register and slice visible wires.
        try:
            samples_new_for_search = ensemble.iqp_circuit.sample(
                trainer.final_params, shots=n_shots_search, wires=ensemble.wires
            )
        except TypeError:
            samples_new_for_search = ensemble.iqp_circuit.sample(
                trainer.final_params, shots=n_shots_search
            )
        if ensemble.wires is not None and samples_new_for_search.shape[1] != len(ensemble.wires):
            samples_new_for_search = samples_new_for_search[:, ensemble.wires]

    key, subkey = jax.random.split(key, 2)
    schedule = config.get('lambda_schedule') or {}
    gamma = float(schedule.get('gamma', 2.0))
    tau = float(schedule.get('tau', 2.0))
    alpha_initial = ensemble.add_model(trainer.final_params, subkey, gamma=gamma, tau=tau)

    # Apply Weight Strategy
    alpha = alpha_initial
    if weight_strategy == 'line_search':
        alpha = ensemble.apply_weight_strategy('line_search',
                                             trs_old=traces.get('trs_ens'),
                                             trs_new=traces.get('trs_iqp'),
                                             trs_data=traces.get('trs_data'),
                                             trs_corr_old=traces.get('trs_corr_ens'),
                                             trs_corr_new=traces.get('trs_corr_iqp'),
                                             ground_truth=x_train)
    elif weight_strategy == 'tvd_line_search':
        alpha = ensemble.apply_weight_strategy('tvd_line_search',
                                             samples_old=samples_old_for_search,
                                             samples_new=samples_new_for_search,
                                             ground_truth=x_train)
    elif weight_strategy == 'validity_line_search':
        alpha = ensemble.apply_weight_strategy('validity_line_search',
                                             samples_old=samples_old_for_search,
                                             samples_new=samples_new_for_search,
                                             validity_fn=validity_fn,
                                             alpha_n_grid=alpha_n_grid)
    elif weight_strategy == 'coverage_line_search':
        alpha = ensemble.apply_weight_strategy('coverage_line_search',
                                             samples_old=samples_old_for_search,
                                             samples_new=samples_new_for_search,
                                             ground_truth=x_train,
                                             coverage_fn=coverage_fn,
                                             alpha_n_grid=alpha_n_grid)
    elif weight_strategy == 'sum_line_search':
        alpha = ensemble.apply_weight_strategy('sum_line_search',
                                             samples_old=samples_old_for_search,
                                             samples_new=samples_new_for_search,
                                             ground_truth=x_train,
                                             validity_fn=validity_fn,
                                             coverage_fn=coverage_fn,
                                             alpha_n_grid=alpha_n_grid,
                                             validity_weight=alpha_objective_weight)
    elif weight_strategy == 'fully_corrective':
        alpha = ensemble.apply_weight_strategy('fully_corrective',
                                             trs_data=traces.get('trs_data'))
    elif weight_strategy in ('greedy', 'frank_wolfe'):
        alpha = ensemble.apply_weight_strategy(weight_strategy)

    ensemble.training_losses.append({
        'total': np.array(trainer.losses),
        'data_hist': np.array(data_hist) if data_hist else None,
        'ens_hist': np.array(ens_hist) if ens_hist else None,
        'hist_epochs': np.array(hist_epochs) if hist_epochs else None,
        'data_final': float(final_comp['data']),
        'ensemble_final': float(final_comp['ensemble']),
    })

    if config.get('verbose', False):
        report_loss_components(final_comp['data'], final_comp['ensemble'], trainer.losses[-1])

    return key, alpha


def run_data_only_ensemble_baseline(
    circuit: iqp.IqpSimulator,
    x_train: np.ndarray,
    key: jax.Array,
    config: dict,
    sigma: float | list,
    n_ops: int,
    n_samples: int,
    wires: list | None,
    monitor_interval: int | None,
    turbo_opt: int | None,
    evaluation: EvaluationPolicy,
    acceptance_metric: str,
    min_alpha_accept: float,
    validity_fn: callable,
    coverage_fn: callable,
) -> tuple:
    """Train a data-only iterative ensemble baseline (lambda_dual=0)."""
    n_models = int(config['n_models'])
    cfg = dict(config)
    cfg['lambda_dual'] = 0.0

    print(f"\nTraining data-only iterative baseline ({n_models} models, lambda_dual=0)...")

    ensemble = BoostedEnsemble(
        circuit,
        n_models=n_models,
        sigma=sigma,
        n_ops=n_ops,
        n_samples=n_samples,
        lambda_dual=0.0,
        wires=wires,
        max_batch_ops=config.get('max_batch_ops', None),
        max_batch_samples=config.get('max_batch_samples', None),
    )

    key, trainer_m0, alpha_0 = train_ensemble_model_0(ensemble, x_train, key, cfg, monitor_interval, turbo_opt)
    m0_training_mmd = evaluation.evaluate_ensemble_training_mmd(ensemble)

    if not evaluation.sampling_enabled:
        ens_stats = evaluation.analytical_mmd_stats(m0_training_mmd)
    else:
        _, ens_stats = evaluation.sample_and_evaluate_ensemble(ensemble, step=0)

    report_step(
        0,
        n_models,
        training_mmd=m0_training_mmd,
        sampled_mmd=ens_stats['mmd'] if evaluation.sampling_enabled else None,
        alpha_opt=alpha_0,
    )

    history = {k: [v] for k, v in ens_stats.items()}
    history['step'] = [0]
    history['alpha'] = [alpha_0]
    history['training_loss'] = [m0_training_mmd]

    prev_ens_stats = ens_stats
    prev_training_mmd = m0_training_mmd

    for step in range(1, n_models):
        print(f"[Data-only Step {step}] Training Model {step}...")
        snapshot = ensemble.snapshot_state()

        key, alpha = train_boosting_step(ensemble, x_train, key, cfg, monitor_interval, turbo_opt,
                                          snapshot, step=step,
                                          validity_fn=validity_fn, coverage_fn=coverage_fn)

        if alpha <= min_alpha_accept:
            print(f"  [Data-only Step {step}] REJECTED (alpha_opt={alpha:.3e} <= {min_alpha_accept:.1e})")
            ensemble.restore_state(snapshot)
            continue

        mixture_training_mmd = evaluation.evaluate_ensemble_training_mmd(ensemble)
        if not evaluation.sampling_enabled:
            ens_stats = evaluation.analytical_mmd_stats(mixture_training_mmd)
        else:
            _, ens_stats = evaluation.sample_and_evaluate_ensemble(ensemble, step=step)

        report_step(
            step,
            n_models,
            training_mmd=mixture_training_mmd,
            sampled_mmd=ens_stats['mmd'] if evaluation.sampling_enabled else None,
            alpha_opt=alpha,
        )

        if acceptance_metric == 'training_mmd':
            current_accept_metric = mixture_training_mmd
            previous_accept_metric = prev_training_mmd
        else:
            current_accept_metric = ens_stats['mmd']
            previous_accept_metric = prev_ens_stats['mmd']

        accepted, should_stop = check_acceptance(
            current_accept_metric,
            previous_accept_metric,
            ensemble,
            snapshot,
            step,
            keep_all=config.get('keep_models_for_diagnosis', False),
            stop_on_reject=config.get('stop_on_reject', False),
        )

        if should_stop:
            break

        if not accepted:
            continue

        for k, v in ens_stats.items():
            if k in history:
                history[k].append(v)
        history['step'].append(step)
        history['alpha'].append(alpha)
        history['training_loss'].append(mixture_training_mmd)

        prev_ens_stats = ens_stats
        prev_training_mmd = mixture_training_mmd

    if not evaluation.final_sampling_enabled:
        final_stats = evaluation.analytical_mmd_stats(
            history['training_loss'][-1] if history['training_loss'] else float('nan')
        )
    else:
        _, final_stats = evaluation.sample_and_evaluate_ensemble(ensemble, step=n_models)

    return key, ensemble, history, final_stats


def _resolve_baselines_to_run(config: dict) -> list[str]:
    """Resolve enabled baselines from a unified config key.

    Supports `baseline` as either a string or list of strings.
    """
    baselines = config.get('baseline', None)
    if isinstance(baselines, str):
        baselines = [baselines]

    # Backward compatibility for older configs
    if baselines is None:
        baselines = config.get('baselines_to_run', None)
    if baselines is None:
        baselines = ['standalone']
        if config.get('run_data_only_baseline', False):
            baselines.append('data_only')
    baselines = [b for b in baselines if b in {'standalone', 'data_only', 'none'}]
    if 'none' in baselines:
        return []
    if not baselines:
        baselines = ['standalone']
    return baselines


def run_boosting_experiment(
    config: dict,
    dataset: DatasetBundle,
    dataset_spec: dict | None,
    metric_configs: list = None,
    baseline_epochs: int | None = None,
    output_base_dir: str = 'out',
    run_name: str | None = None,
    log_dir: str | None = None,
    log_filename: str = 'log.txt',
    append_log: bool = False,
):
    """Run a complete ensemble boosting experiment."""
    np.random.seed(config['rng_seed'])
    x_train = dataset.x_train
    validity_fn = dataset.validity_fn
    coverage_fn = dataset.coverage_fn

    output = OutputManager(
        base_dir=output_base_dir,
        run_name=run_name,
        log_dir=log_dir,
        log_filename=log_filename,
        append_log=append_log,
    )

    with output:
        report_config(config, dataset.dataset_name)
        output.save_config(config)

        n_qubits = dataset.n_qubits

        # Circuit setup
        circuit_config = config.get('circuit_config', {'topology': 'neighbour', 'distance': 3, 'max_weight': 2})
        circuit_kwargs = dict(circuit_config)
        circuit_kwargs.setdefault('data', x_train)
        circuit, gates, gate_desc, wires = setup_iqp_circuit(n_qubits, **circuit_kwargs)
        report_circuit(gate_desc)
        save_circuit_plot(circuit, output)

        # Sigma setup from config (supports median, percentile, medoids)
        sigma = compute_sigma(config, x_train, seed=config.get('data_seed', 42))

        n_ops = config.get('n_ops', 1000)
        n_samples = int(config.get('n_samples', 1000))
        shots = int(config.get('shots', 1000))

        report_kernel(sigma, n_ops, n_qubits)

        key = jax.random.PRNGKey(config['rng_seed'])
        rng_seed = int(config['rng_seed'])

        plot_cfg = get_plot_config()
        monitor_interval = plot_cfg['plot_interval'] if plot_cfg['plot_data_loss'] else None
        turbo_opt = config.get('turbo', None)
        evaluation = dataset.build_evaluation_policy(
            sigma=sigma,
            shots=shots,
            rng_seed=rng_seed,
            skip_sampling=config.get('skip_sampling', False),
            final_eval_sampling=bool(config.get('final_eval_sampling', False)),
        )
        min_alpha_accept = float(config.get('min_alpha_accept', 1e-10))
        acceptance_metric = config.get(
            'acceptance_metric',
            'sample_mmd' if evaluation.sampling_enabled else 'training_mmd'
        )
        if acceptance_metric not in {'sample_mmd', 'training_mmd'}:
            raise ValueError("acceptance_metric must be 'sample_mmd' or 'training_mmd'")
        if acceptance_metric == 'sample_mmd' and not evaluation.sampling_enabled:
            acceptance_metric = 'training_mmd'

        if config.get('weight_strategy', 'frank_wolfe') == 'frank_wolfe':
            lambda_sched = config.get('lambda_schedule') or {}
            if lambda_sched.get('type', 'frank_wolfe') != 'frank_wolfe':
                print("\n[WARNING] Causal Mismatch Detected!")
                print("weight_strategy is 'frank_wolfe', but lambda_schedule type is not.")
                print("To ensure second-order equivalence, lambda_schedule will be forced to 'frank_wolfe'.")
                config['lambda_schedule'] = {'type': 'frank_wolfe', 'gamma': 1.0, 'tau': 1.0}


        baselines_to_run = _resolve_baselines_to_run(config)
        print(f"Baselines enabled: {', '.join(baselines_to_run)}")
        standalone_stats = None
        baseline_train_losses = None
        baseline_samples = None
        baseline_params = None
        data_only_stats = None
        data_only_history = None
        data_only_ensemble = None

        # 1. Optional standalone baseline
        if 'standalone' in baselines_to_run:
            if baseline_epochs is None:
                baseline_epochs = config['epochs_per_step'] * config['n_models']

            print(f"\nTraining standalone baseline model ({baseline_epochs} epochs)...")
            key, trainer_base, baseline_params = train_standalone_model(
                circuit, x_train, key, sigma, n_ops, n_samples, config,
                baseline_epochs, monitor_interval, turbo_opt, wires=wires
            )

            baseline_train_losses = getattr(trainer_base, "losses", [])
            baseline_final_loss = float(baseline_train_losses[-1]) if len(baseline_train_losses) > 0 else float('nan')

            if not evaluation.final_sampling_enabled:
                print(f"  [skip_sampling=True] Skipping baseline state vector evaluation. Using training loss: {baseline_final_loss:.6f}")
                baseline_samples = None
                standalone_stats = {'mmd': baseline_final_loss, 'training_loss': baseline_final_loss}
                report_baseline(baseline_final_loss, standalone_stats)
            else:
                baseline_samples, standalone_stats = evaluation.sample_and_evaluate_circuit(
                    circuit, baseline_params, wires=wires
                )
                standalone_stats['training_loss'] = baseline_final_loss
                print()
                report_baseline(standalone_stats['mmd'], standalone_stats)
                print(f"  (Analytical baseline MMD from training: {baseline_final_loss:.6f})")

        # 2. Optional data-only iterative baseline
        if 'data_only' in baselines_to_run:
            key, data_only_ensemble, data_only_history, data_only_stats = run_data_only_ensemble_baseline(
                circuit=circuit,
                x_train=x_train,
                key=key,
                config=config,
                sigma=sigma,
                n_ops=n_ops,
                n_samples=n_samples,
                wires=wires,
                monitor_interval=monitor_interval,
                turbo_opt=turbo_opt,
                evaluation=evaluation,
                acceptance_metric=acceptance_metric,
                min_alpha_accept=min_alpha_accept,
                validity_fn=validity_fn,
                coverage_fn=coverage_fn,
            )
            print(f"Data-only iterative baseline final MMD={data_only_stats['mmd']:.4f}")
            report_baseline(data_only_stats['mmd'], data_only_stats)

        # 3. Initialize Ensemble
        ensemble = BoostedEnsemble(circuit, n_models=config['n_models'],
            sigma=sigma, n_ops=n_ops, n_samples=n_samples,
            lambda_dual=config.get('lambda_dual', 1.0), wires=wires,
            max_batch_ops=config.get('max_batch_ops', None),
            max_batch_samples=config.get('max_batch_samples', None)
        )

        # 4. Train Ensemble Model 0
        print(f"\nTraining ensemble Model 0 ({config['epochs_per_step']} epochs)...")
        key, trainer_m0, alpha_0 = train_ensemble_model_0(ensemble, x_train, key, config, monitor_interval, turbo_opt)

        m0_losses = getattr(trainer_m0, "losses", [])
        if len(m0_losses) > 0:
            print(f"Model 0 final loss: {float(m0_losses[-1]):.6f}")

        # Initial Ensemble Evaluation
        m0_training_mmd = evaluation.evaluate_ensemble_training_mmd(ensemble)
        if not evaluation.sampling_enabled:
            ens_samples = None
            ens_stats = evaluation.analytical_mmd_stats(m0_training_mmd)
        else:
            ens_samples, ens_stats = evaluation.sample_and_evaluate_ensemble(ensemble, step=0)

        report_step(
            0,
            config['n_models'],
            training_mmd=m0_training_mmd,
            sampled_mmd=ens_stats['mmd'] if evaluation.sampling_enabled else None,
            sampled_tvd=ens_stats.get('tvd') if evaluation.sampling_enabled else None,
            alpha_opt=alpha_0,
            oracle_tvd=dataset.top_k_tvd_fn(1) if dataset.top_k_tvd_fn else None,
        )
        prev_ens_stats = ens_stats
        prev_training_mmd = m0_training_mmd

        # Metrics history
        ensemble_metrics_history = {k: [v] for k, v in ens_stats.items()}
        ensemble_metrics_history['step'] = [0]
        ensemble_metrics_history['alpha'] = [alpha_0]
        ensemble_metrics_history['training_loss'] = [m0_training_mmd]

        # 5. Boosting Loop (Model 1, 2, ...)
        for step in range(1, config['n_models']):
            print(f"\n[Step {step}] Training Model {step}...")

            # The effective step for scheduling is the number of models currently in the ensemble.
            # This ensures that if a model is rejected, lambda doesn't prematurely decay.
            effective_step = len(ensemble.models)
            
            # Apply lambda annealing schedule if configured
            current_lambda = compute_lambda_schedule(
                effective_step, config['n_models'],
                base_lambda=config.get('lambda_dual', 1.0),
                schedule=config.get('lambda_schedule', None),
            )
            if current_lambda != ensemble.lambda_dual:
                ensemble.lambda_dual = current_lambda
                print(f"  lambda_dual: {current_lambda:.4f}")

            snapshot = ensemble.snapshot_state()

            key, alpha = train_boosting_step(ensemble, x_train, key, config, monitor_interval, turbo_opt,
                                            snapshot, step=step, validity_fn=validity_fn, coverage_fn=coverage_fn)

            # Zero-weight models do not change the mixture; reject early to avoid
            # sample-noise acceptance and empty per-model sample rows.
            if alpha <= min_alpha_accept:
                print(f"  [Step {step}] REJECTED (alpha_opt={alpha:.3e} <= {min_alpha_accept:.1e})")
                ensemble.restore_state(snapshot)
                continue

            # Evaluate ensemble
            mixture_training_mmd = evaluation.evaluate_ensemble_training_mmd(ensemble)

            if not evaluation.sampling_enabled:
                ens_samples = None
                ens_stats = evaluation.analytical_mmd_stats(mixture_training_mmd)
            else:
                ens_samples, ens_stats = evaluation.sample_and_evaluate_ensemble(ensemble, step=step)

            report_step(
                step,
                config['n_models'],
                training_mmd=mixture_training_mmd,
                sampled_mmd=ens_stats['mmd'] if evaluation.sampling_enabled else None,
                sampled_tvd=ens_stats.get('tvd') if evaluation.sampling_enabled else None,
                alpha_opt=alpha,
                oracle_tvd=dataset.top_k_tvd_fn(step + 1) if dataset.top_k_tvd_fn else None,
            )

            # Check acceptance using configured comparison metric
            if acceptance_metric == 'training_mmd':
                current_accept_metric = mixture_training_mmd
                previous_accept_metric = prev_training_mmd
            else:
                current_accept_metric = ens_stats['mmd']
                previous_accept_metric = prev_ens_stats['mmd']

            accepted, should_stop = check_acceptance(
                current_accept_metric, previous_accept_metric, ensemble, snapshot, step,
                keep_all=config.get('keep_models_for_diagnosis', False),
                stop_on_reject=config.get('stop_on_reject', False)
            )

            if should_stop:
                break

            if not accepted:
                continue

            for k, v in ens_stats.items():
                if k in ensemble_metrics_history:
                    ensemble_metrics_history[k].append(v)
            ensemble_metrics_history['step'].append(step)
            ensemble_metrics_history['alpha'].append(alpha)
            ensemble_metrics_history['training_loss'].append(mixture_training_mmd)

            # SNR computation
            try:
                key, snr_key = jax.random.split(key)
                new_params = ensemble.models[-1]
                snr_mmd_samples = int(min(0.1 * ensemble.n_samples, 100))
                # Build truncated EnsembleTerms with only previous models
                snr_terms = EnsembleTerms()
                snr_terms.trs = ensemble.terms.trs[:-1]
                snr_terms.corrs = ensemble.terms.corrs[:-1]
                snr_terms.ops = ensemble.terms.ops
                def dual_loss_for_snr(params, iqp_circuit, x_train, key, sigma, n_ops):
                    return dual_mmd_loss(
                        params, iqp_circuit, x_train, snr_terms, ensemble.weights[:-1],
                        sigma, n_ops, snr_mmd_samples, key,
                        lambda_dual=ensemble.lambda_dual,
                        stochastic_ops=False,
                        wires=ensemble.wires,
                        max_batch_ops=ensemble.max_batch_ops,
                        max_batch_samples=ensemble.max_batch_samples
                    )
                snr_info = gradient_snr(
                    new_params, ensemble.iqp_circuit, x_train,
                    dual_loss_for_snr, key=snr_key,
                    sigma=ensemble.sigma, n_ops=ensemble.n_ops
                )
                report_gradient_snr(snr_info)
            except Exception as e:
                print(f"  [Debug] Gradient SNR computation failed: {str(e)[:80]}")

            prev_ens_stats = ens_stats
            prev_training_mmd = mixture_training_mmd
            gc.collect()
            if step % 2 == 0:
                jax.clear_caches()

        baseline_catalog = {
            'standalone': ('Standalone', standalone_stats),
            'data_only': ('Data-only', data_only_stats),
        }
        reference_label, reference_stats = 'Synthetic', {'mmd': m0_training_mmd}
        for b in baselines_to_run:
            label, stats = baseline_catalog.get(b, (None, None))
            if stats is not None:
                reference_label, reference_stats = label, stats
                print(f"Baseline used as reference: {reference_label}")
                break

        final_evaluation = run_final_evaluation(FinalEvaluationContext(
            config=config,
            dataset=dataset,
            output=output,
            ensemble=ensemble,
            data_only_ensemble=data_only_ensemble,
            data_only_stats=data_only_stats,
            data_only_history=data_only_history,
            standalone_stats=standalone_stats,
            baseline_train_losses=baseline_train_losses,
            ensemble_metrics_history=ensemble_metrics_history,
            evaluation=evaluation,
            sigma=sigma,
            shots=shots,
            rng_seed=rng_seed,
            reference_stats=reference_stats,
            reference_label=reference_label,
            metric_configs=metric_configs,
        ))

        # Custom Visualization -- always attempt if a viz callback is set.
        # The viz function handles None samples gracefully (e.g. Ising Lorenz).
        if dataset.custom_viz_fn is not None:
            try:
                dataset.run_custom_visualization(
                    output,
                    baseline_samples if evaluation.final_sampling_enabled else None,
                    final_evaluation.final_ensemble_samples if evaluation.final_sampling_enabled else None,
                    final_evaluation.per_model_samples if evaluation.final_sampling_enabled else [],
                    ensemble.weights,
                )
            except Exception as e:
                print(f"Custom visualization failed: {e}")

        # Save circuit artifact for backend execution if requested
        if config.get('save_circuit_artifacts', False):
            print("\n[ARTIFACTS] Saving circuit artifact for backend execution...")
            try:
                if save_circuit_artifact is None:
                    raise ImportError("src.circuit_artifacts is not available")
                artifact_path = save_circuit_artifact(
                    path=output.get_path('circuit_artifact.json'),
                    dataset_name=dataset.dataset_name,
                    run_name=run_name,
                    config=config,
                    dataset_spec=dataset_spec,
                    x_train=x_train,
                    sigma=sigma,
                    circuit=circuit,
                    circuit_config=circuit_config,
                    n_visible_qubits=n_qubits,
                    wires=wires,
                    ensemble=ensemble,
                    ensemble_metrics_history=ensemble_metrics_history,
                    ensemble_fcfw_weights=final_evaluation.ensemble_fcfw_weights,
                    standalone_params=baseline_params,
                    data_only_ensemble=data_only_ensemble,
                    data_only_history=data_only_history,
                    data_only_fcfw_weights=final_evaluation.data_only_fcfw_weights,
                )
                print(f"[ARTIFACTS] Circuit artifact saved to: {artifact_path}")
            except Exception as e:
                print(f"[ARTIFACTS] Failed to save circuit artifact: {e}")

        return {
            'final_stats': final_evaluation.final_stats,
            'ensemble_metrics_history': ensemble_metrics_history,
            'ensemble': ensemble,
            'baseline_stats': standalone_stats,
            'data_only_stats': data_only_stats,
            'ensemble_fcfw_stats': final_evaluation.ensemble_fcfw_stats,
            'ensemble_fcfw_weights': final_evaluation.ensemble_fcfw_weights,
            'output_dir': str(output.run_dir),
            'weights': np.asarray(ensemble.weights, dtype=np.float64),
            'n_models_accepted': len(ensemble.models),
        }
