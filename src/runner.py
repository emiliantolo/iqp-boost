"""Common runner for ensemble boosting experiments."""

import iqpopt as iqp
from iqpopt.gen_qml.utils import median_heuristic
from iqpopt.gen_qml.iqp_methods import mmd_loss_iqp
from src.ensemble import BoostedEnsemble
from src.reporting import (
    report_metrics_table, get_plot_config, OutputManager, plot_data_ensemble_loss,
    plot_metrics_progression, report_baseline, report_final, report_rejection,
    report_step, report_gradient_snr, report_config, report_circuit, report_kernel,
    report_loss_components, report_acceptance, save_circuit_plot
)
from src.core import (
    setup_iqp_circuit, get_params_init
)
from src.utils import compute_mmd, compute_kl_divergence, compute_metrics, compute_precision_recall_f1, compute_jsd, compute_tvd
from src.dual_mmd_loss import gradient_snr, dual_mmd_loss, EnsembleTerms
import jax
import numpy as np
import gc


def evaluate_samples(ground_truth: np.ndarray, samples: np.ndarray, sigma: float | list,
                     validity_fn: callable, coverage_fn: callable) -> dict:
    """Evaluate all metrics: MMD, KL, validity, coverage, precision, recall, F1."""
    mmd = compute_mmd(ground_truth, samples, sigma)

    # Use the first sigma for precision computation
    sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
    precision_sigma = sigmas[0]

    metrics = compute_metrics(ground_truth, samples, validity_fn, coverage_fn)
    prf_metrics = compute_precision_recall_f1(ground_truth, samples, precision_sigma)
    kl = compute_kl_divergence(ground_truth, samples)
    jsd = compute_jsd(ground_truth, samples)
    tvd = compute_tvd(ground_truth, samples)

    return {
        'mmd': mmd,
        'kl': kl,
        'jsd': jsd,
        'tvd': tvd,
        'validity': metrics['validity_rate'],
        'coverage': metrics['coverage'],
        'precision': prf_metrics['precision'],
        'recall': prf_metrics['recall'],
        'support_match': prf_metrics['support_match'],
        'f_score': prf_metrics['f_score'],
    }


def compute_ensemble_training_mmd(ensemble: BoostedEnsemble, ground_truth: np.ndarray) -> float:
    """Compute analytical ensemble MMD^2 wrt data using cached trace estimates."""
    if not ensemble.models or not ensemble.weights:
        return float('nan')

    n_samples = ensemble.n_samples
    m = len(ground_truth)
    n_sigmas = len(ensemble.terms.ops)
    if n_sigmas == 0:
        return float('nan')

    weights = np.asarray(ensemble.weights, dtype=float)
    mmd_vals = []

    for sigma_idx, (_, visible_ops) in ensemble.terms.ops.items():
        tr_data = np.mean(1 - 2 * ((ground_truth @ np.asarray(visible_ops).T) % 2), axis=0)

        tr_enss = np.asarray([np.asarray(t[sigma_idx]) for t in ensemble.terms.trs])
        corr_enss = np.asarray([np.asarray(c[sigma_idx]) for c in ensemble.terms.corrs])

        tr_mix = np.sum(weights[:, None] * tr_enss, axis=0)
        tr_mix_sq = np.einsum('i,ik,jk,j->k', weights, tr_enss, tr_enss, weights)
        corr_mix = np.sum((weights**2)[:, None] * corr_enss, axis=0)

        term_mix_mix = np.mean((tr_mix_sq - corr_mix) * n_samples / (n_samples - 1))
        term_mix_data = np.mean(tr_mix * tr_data)
        term_data_data = np.mean((tr_data * tr_data * m - 1) / (m - 1))

        mmd_vals.append(term_mix_mix - 2.0 * term_mix_data + term_data_data)

    return float(np.mean(mmd_vals))


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
    alpha = ensemble.add_model(trainer.final_params, subkey)

    ensemble.training_losses.append({
        'total': np.array(trainer.losses),
        'data_final': None,
        'ensemble_final': None
    })

    return key, trainer, alpha


def train_boosting_step(ensemble: BoostedEnsemble, x_train: np.ndarray, key: jax.Array,
                        config: dict, monitor_interval: int | None, turbo_opt: int | None,
                        snapshot: dict, rng: np.random.Generator) -> tuple:
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
        max_batch_ops=ensemble.max_batch_ops, max_batch_samples=ensemble.max_batch_samples
    )

    final_comp = compute_dual_components_from_traces(
        traces,
        n_samples=ensemble.n_samples,
        n_data=len(x_train),
    )

    key, subkey = jax.random.split(key, 2)
    alpha_initial = ensemble.add_model(trainer.final_params, subkey)

    # Apply Weight Strategy
    alpha = alpha_initial
    weight_strategy = config.get('weight_strategy', 'greedy')
    if weight_strategy == 'line_search':
        alpha = ensemble.apply_weight_strategy('line_search',
                                             trs_old=traces.get('trs_ens'),
                                             trs_new=traces.get('trs_iqp'),
                                             trs_data=traces.get('trs_data'),
                                             trs_corr_old=traces.get('trs_corr_ens'),
                                             trs_corr_new=traces.get('trs_corr_iqp'),
                                             ground_truth=x_train)
    elif weight_strategy == 'fully_corrective':
        alpha = ensemble.apply_weight_strategy('fully_corrective',
                                             trs_data=traces.get('trs_data'))

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
    shots: int,
    wires: list | None,
    monitor_interval: int | None,
    turbo_opt: int | None,
    skip_sampling: bool,
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

    # Use the same sampling RNG policy as the boosting run for comparable evaluation noise.
    rng = np.random.default_rng(0)

    key, trainer_m0, alpha_0 = train_ensemble_model_0(ensemble, x_train, key, cfg, monitor_interval, turbo_opt)
    m0_training_mmd = compute_ensemble_training_mmd(ensemble, x_train)

    if skip_sampling:
        ens_stats = {'mmd': m0_training_mmd}
    else:
        ens_samples = ensemble.sample(shots, rng)
        ens_stats = evaluate_samples(x_train, ens_samples, sigma, validity_fn, coverage_fn)

    history = {k: [v] for k, v in ens_stats.items()}
    history['step'] = [0]
    history['alpha'] = [alpha_0]
    history['training_loss'] = [m0_training_mmd]

    prev_ens_stats = ens_stats
    prev_training_mmd = m0_training_mmd

    for step in range(1, n_models):
        print(f"[Data-only Step {step}] Training Model {step}...")
        snapshot = ensemble.snapshot_state()

        key, alpha = train_boosting_step(ensemble, x_train, key, cfg, monitor_interval, turbo_opt, snapshot, rng)

        if alpha <= min_alpha_accept:
            print(f"  [Data-only Step {step}] REJECTED (alpha={alpha:.3e} <= {min_alpha_accept:.1e})")
            ensemble.restore_state(snapshot)
            continue

        mixture_training_mmd = compute_ensemble_training_mmd(ensemble, x_train)
        if skip_sampling:
            ens_stats = {'mmd': mixture_training_mmd}
        else:
            ens_samples = ensemble.sample(shots, rng)
            ens_stats = evaluate_samples(x_train, ens_samples, sigma, validity_fn, coverage_fn)

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

    if skip_sampling:
        final_stats = {'mmd': history['training_loss'][-1] if history['training_loss'] else float('nan')}
    else:
        final_samples = ensemble.sample(shots, rng)
        final_stats = evaluate_samples(x_train, final_samples, sigma, validity_fn, coverage_fn)

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
    baselines = [b for b in baselines if b in {'standalone', 'data_only'}]
    if not baselines:
        baselines = ['standalone']
    return baselines


def run_boosting_experiment(
    config: dict,
    dataset_name: str,
    x_train: np.ndarray,
    validity_fn: callable,
    coverage_fn: callable,
    custom_viz_fn: callable = None,
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

    output = OutputManager(
        base_dir=output_base_dir,
        run_name=run_name,
        log_dir=log_dir,
        log_filename=log_filename,
        append_log=append_log,
    )

    with output:
        report_config(config, dataset_name)
        output.save_config(config)

        n_qubits = x_train.shape[1]

        # Circuit setup
        circuit_config = config.get('circuit_config', {'topology': 'neighbour', 'distance': 3, 'max_weight': 2})
        circuit, gates, gate_desc, wires = setup_iqp_circuit(n_qubits, **circuit_config)
        report_circuit(gate_desc)
        save_circuit_plot(circuit, output)

        # Sigma setup from config
        sigma_base = median_heuristic(x_train)
        print(f"sigma_base={sigma_base:.8f}")

        n_ops = config.get('n_ops', 1000)
        n_samples = int(config.get('n_samples', 1000))
        shots = int(config.get('shots', 1000))

        # Build sigma: either from sigma_factor(s) or direct sigma value(s)
        if 'sigma_factor' in config:
            sf = config['sigma_factor']
            if isinstance(sf, list):
                sigma = [s * sigma_base for s in sf]
            else:
                sigma = sf * sigma_base
        elif 'sigma' in config:
            sigma = config['sigma']
        else:
            sigma = 0.4 * sigma_base  # default

        report_kernel(sigma, n_ops, n_qubits)

        key = jax.random.PRNGKey(config['rng_seed'])

        plot_cfg = get_plot_config()
        monitor_interval = plot_cfg['plot_interval'] if plot_cfg['plot_data_loss'] else None
        turbo_opt = config.get('turbo', None)

        rng = np.random.default_rng(0)
        skip_sampling = config.get('skip_sampling', False)
        sampling_enabled = not skip_sampling
        min_alpha_accept = float(config.get('min_alpha_accept', 1e-10))
        acceptance_metric = config.get(
            'acceptance_metric',
            'sample_mmd' if sampling_enabled else 'training_mmd'
        )
        if acceptance_metric not in {'sample_mmd', 'training_mmd'}:
            raise ValueError("acceptance_metric must be 'sample_mmd' or 'training_mmd'")
        if acceptance_metric == 'sample_mmd' and not sampling_enabled:
            acceptance_metric = 'training_mmd'

        baselines_to_run = _resolve_baselines_to_run(config)
        print(f"Baselines enabled: {', '.join(baselines_to_run)}")
        standalone_stats = None
        baseline_train_losses = None
        baseline_samples = None
        data_only_stats = None
        data_only_history = None

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

            if skip_sampling:
                print(f"  [skip_sampling=True] Skipping baseline state vector evaluation. Using training loss: {baseline_final_loss:.6f}")
                baseline_samples = None
                standalone_stats = {'mmd': baseline_final_loss, 'training_loss': baseline_final_loss}
                report_baseline(baseline_final_loss, standalone_stats)
            else:
                baseline_samples = circuit.sample(baseline_params, shots=shots)
                if wires is not None:
                    baseline_samples = baseline_samples[:, wires]
                standalone_stats = evaluate_samples(x_train, baseline_samples, sigma, validity_fn, coverage_fn)
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
                shots=shots,
                wires=wires,
                monitor_interval=monitor_interval,
                turbo_opt=turbo_opt,
                skip_sampling=skip_sampling,
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
        m0_training_mmd = compute_ensemble_training_mmd(ensemble, x_train)
        if skip_sampling:
            ens_samples = None
            ens_stats = {'mmd': m0_training_mmd}
        else:
            ens_samples = ensemble.sample(shots, rng)
            ens_stats = evaluate_samples(x_train, ens_samples, sigma, validity_fn, coverage_fn)

        report_step(
            0,
            config['n_models'],
            training_mmd=m0_training_mmd,
            sampled_mmd=ens_stats['mmd'] if sampling_enabled else None,
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
            snapshot = ensemble.snapshot_state()

            key, alpha = train_boosting_step(ensemble, x_train, key, config, monitor_interval, turbo_opt,
                                            snapshot, rng)

            # Zero-weight models do not change the mixture; reject early to avoid
            # sample-noise acceptance and empty per-model sample rows.
            if alpha <= min_alpha_accept:
                print(f"  [Step {step}] REJECTED (alpha={alpha:.3e} <= {min_alpha_accept:.1e})")
                ensemble.restore_state(snapshot)
                continue

            # Evaluate ensemble
            mixture_training_mmd = compute_ensemble_training_mmd(ensemble, x_train)

            if skip_sampling:
                ens_samples = None
                ens_stats = {'mmd': mixture_training_mmd}
            else:
                ens_samples = ensemble.sample(shots, rng)
                ens_stats = evaluate_samples(x_train, ens_samples, sigma, validity_fn, coverage_fn)

            report_step(
                step,
                config['n_models'],
                training_mmd=mixture_training_mmd,
                sampled_mmd=ens_stats['mmd'] if sampling_enabled else None,
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
                        lambda_dual=config['lambda_dual'],
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

        # Final reporting — sample once, reuse everywhere
        if skip_sampling:
            print("\n[skip_sampling=True] Skipping final state vector sampling. Using final training losses.")
            final_ensemble_samples, final_counts, per_model_samples = None, None, []
            
            # Use last ensemble loss as final stats
            final_loss = ensemble_metrics_history['training_loss'][-1] if ensemble_metrics_history['training_loss'] else float('nan')
            final_stats = {'mmd': final_loss}
            
            report_stats = {**final_stats}
            report_final(reference_stats['mmd'], final_stats['mmd'], len(ensemble.models), report_stats)
            
            # Simplified metrics table (only MMD)
            model_rows = []
            if data_only_stats is not None and reference_label != 'Data-only':
                model_rows.append(("Data-only", {'mmd': data_only_stats.get('mmd', float('nan'))}))
            for i in range(len(ensemble.models)):
                # Try to get training loss at each step if recorded
                m_loss = ensemble_metrics_history['training_loss'][i] if i < len(ensemble_metrics_history['training_loss']) else float('nan')
                model_rows.append((f"Model {i}", {'mmd': m_loss}))
            
            report_metrics_table(reference_stats, final_stats, model_rows, "FINAL MODEL COMPARISON (Analytical)")
            output.save_results_csv(ensemble_metrics_history, baseline_stats=reference_stats)
        else:
            final_ensemble_samples, final_counts, per_model_samples = ensemble.sample(
                shots, rng, return_details=True)
            final_stats = evaluate_samples(x_train, final_ensemble_samples, sigma, validity_fn, coverage_fn)

            report_final(reference_stats['mmd'], final_stats['mmd'], len(ensemble.models), final_stats)

            # Per-model metrics from existing samples (no re-sampling)
            model_rows = []
            if data_only_stats is not None and reference_label != 'Data-only':
                model_rows.append(("Data-only", data_only_stats))
            for i, model_samples in enumerate(per_model_samples):
                if len(model_samples) > 0:
                    model_stats = evaluate_samples(x_train, model_samples, sigma, validity_fn, coverage_fn)
                else:
                    model_stats = {'mmd': float('nan')}
                model_rows.append((f"Model {i}", model_stats))

            report_metrics_table(reference_stats, final_stats, model_rows, "FINAL MODEL COMPARISON")
            output.save_results_csv(ensemble_metrics_history, baseline_stats=reference_stats)

        # Plotting
        if get_plot_config()['plot_data_loss']:
            plot_data_ensemble_loss(
                ensemble,
                ensemble_metrics_history,
                reference_stats,
                output,
                baseline_train_losses=np.array(baseline_train_losses) if baseline_train_losses is not None else None,
                data_only_history=data_only_history,
            )

            if sampling_enabled:
                if metric_configs is None:
                    metric_configs = [
                        ('mmd', 'Sampled MMD²', 1, 'blue', 's'),
                        ('tvd', 'TVD', 1, 'green', '^'),
                        ('coverage', 'Coverage (%)', 100, 'purple', 'v'),
                        ('validity', 'Validity (%)', 100, 'orange', 'd'),
                    ]
                plot_metrics_progression(ensemble_metrics_history, reference_stats, output, metric_configs)

        # Custom Visualization — pass pre-sampled data to avoid redundant sampling
        if not skip_sampling and custom_viz_fn is not None:
            try:
                custom_viz_fn(output, x_train, baseline_samples, final_ensemble_samples,
                              per_model_samples, ensemble.weights)
            except Exception as e:
                print(f"Custom visualization failed: {e}")

