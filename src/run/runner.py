"""Common runner for ensemble boosting experiments."""

from src.core import BoostedEnsemble, setup_iqp_circuit
from src.core.sigma_heuristics import compute_sigma
try:
    from src.circuit_artifacts import save_circuit_artifact
except ModuleNotFoundError:
    save_circuit_artifact = None
from src.io.reporting import (
    get_plot_config, OutputManager, report_config,
    report_circuit, report_kernel, save_circuit_plot
)
from src.datasets import DatasetBundle
from src.datasets.boltzmann_plots import plot_lorenz_curve
from src.run.baseline import BaselineContext, run_baselines
from src.run.boosting_step import (
    BoostingStepContext,
    initialize_boosting_ensemble,
    run_boosting_step,
)
from src.run.final_evaluation import FinalEvaluationContext, run_final_evaluation
import jax
import numpy as np


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
    skip_plots: bool = True,
):
    """Run a complete ensemble boosting experiment."""
    np.random.seed(config['rng_seed'])
    x_train = dataset.x_train

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
        if not skip_plots:
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
            exact_metrics=config,
        )

        require_exact_sampling = bool(config.get('require_exact_sampling', False))
        exact_sampling = bool(config.get('exact_sampling', False))
        if require_exact_sampling and not exact_sampling:
            raise ValueError("require_exact_sampling=True requires exact_sampling=True")
        if exact_sampling and n_qubits > 20:
            if require_exact_sampling:
                raise ValueError("Exact sampling is required but n_qubits > 20")
            print("  [exact_sampling=True but n_qubits > 20, falling back to sampled metrics]")
            exact_sampling = False
        if exact_sampling and circuit.bitflip:
            if require_exact_sampling:
                raise ValueError("Exact sampling is required but circuit is bitflip mode")
            print("  [exact_sampling=True but circuit is bitflip mode, falling back to sampled metrics]")
            exact_sampling = False

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


        baseline_result = run_baselines(
            BaselineContext(
                config=config,
                circuit=circuit,
                dataset=dataset,
                x_train=x_train,
                key=key,
                sigma=sigma,
                n_ops=n_ops,
                n_samples=n_samples,
                wires=wires,
                monitor_interval=monitor_interval,
                turbo_opt=turbo_opt,
                evaluation=evaluation,
                acceptance_metric=acceptance_metric,
                min_alpha_accept=min_alpha_accept,
                baseline_epochs=baseline_epochs,
            )
        )
        key = baseline_result.key

        # 3. Initialize Ensemble
        ensemble = BoostedEnsemble(circuit, n_models=config['n_models'],
            sigma=sigma, n_ops=n_ops, n_samples=n_samples,
            lambda_dual=config.get('lambda_dual', 1.0), wires=wires,
            max_batch_ops=config.get('max_batch_ops', None),
            max_batch_samples=config.get('max_batch_samples', None)
        )

        # 4. Train Ensemble Model 0
        print(f"\nTraining ensemble Model 0 ({config['epochs_per_step']} epochs)...")
        initial = initialize_boosting_ensemble(
            ensemble=ensemble,
            x_train=x_train,
            key=key,
            config=config,
            monitor_interval=monitor_interval,
            turbo_opt=turbo_opt,
            evaluation=evaluation,
            top_k_tvd_fn=dataset.top_k_tvd_fn,
        )
        key = initial.key
        prev_ens_stats = initial.stats
        prev_training_mmd = initial.training_mmd
        m0_training_mmd = initial.training_mmd
        ensemble_metrics_history = initial.metrics_history

        # 5. Boosting Loop (Model 1, 2, ...)
        for step in range(1, config['n_models']):
            print(f"\n[Step {step}] Training Model {step}...")
            step_result = run_boosting_step(
                BoostingStepContext(
                    config=config,
                    ensemble=ensemble,
                    x_train=x_train,
                    key=key,
                    evaluation=evaluation,
                    metrics_history=ensemble_metrics_history,
                    step=step,
                    monitor_interval=monitor_interval,
                    turbo_opt=turbo_opt,
                    acceptance_metric=acceptance_metric,
                    min_alpha_accept=min_alpha_accept,
                    previous_stats=prev_ens_stats,
                    previous_training_mmd=prev_training_mmd,
                    validity_fn=dataset.validity_fn,
                    coverage_fn=dataset.coverage_fn,
                    top_k_tvd_fn=dataset.top_k_tvd_fn,
                )
            )
            key = step_result.key

            if step_result.should_stop:
                break

            prev_ens_stats = step_result.previous_stats
            prev_training_mmd = step_result.previous_training_mmd

        reference_label, reference_stats = baseline_result.reference_or_synthetic(m0_training_mmd)

        final_evaluation = run_final_evaluation(
            FinalEvaluationContext(
                config=config,
                dataset=dataset,
                output=output,
                ensemble=ensemble,
                data_only_ensemble=baseline_result.data_only_ensemble,
                data_only_stats=baseline_result.data_only_stats,
                data_only_history=baseline_result.data_only_history,
                standalone_stats=baseline_result.standalone_stats,
                baseline_train_losses=baseline_result.standalone_train_losses,
                ensemble_metrics_history=ensemble_metrics_history,
                evaluation=evaluation,
                sigma=sigma,
                shots=shots,
                rng_seed=rng_seed,
                reference_stats=reference_stats,
                reference_label=reference_label,
                metric_configs=metric_configs,
            ),
            skip_plots=skip_plots,
        )

        # Custom Visualization -- always attempt if a viz callback is set.
        # The viz function handles None samples gracefully (e.g. Ising Lorenz).
        if not skip_plots and dataset.custom_viz_fn is not None:
            try:
                dataset.run_custom_visualization(
                    output,
                    baseline_result.standalone_samples if evaluation.final_sampling_enabled else None,
                    final_evaluation.final_ensemble_samples if evaluation.final_sampling_enabled else None,
                    final_evaluation.per_model_samples if evaluation.final_sampling_enabled else [],
                    ensemble.weights,
                )
            except Exception as e:
                print(f"Custom visualization failed: {e}")

        # Global Lorenz curve plot (controlled by plot config)
        if not skip_plots and evaluation.final_sampling_enabled and get_plot_config()['plot_lorenz_curve']:
            try:
                _fig = plot_lorenz_curve(
                    dataset.exact_probs,
                    baseline_result.standalone_samples if evaluation.final_sampling_enabled else None,
                    final_evaluation.final_ensemble_samples,
                    reference_samples=dataset.x_train,
                )
                _path = output.get_path('lorenz_curve.png')
                _fig.savefig(_path, dpi=160, bbox_inches='tight')
                _fig.savefig(str(_path).replace('.png', '.pdf'), bbox_inches='tight')
                print(f"  [lorenz_curve] Saved to {_path}")
                _fig.clf()
            except Exception as e:
                print(f"  [lorenz_curve] Failed: {e}")

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
                    standalone_params=baseline_result.standalone_params,
                    data_only_ensemble=baseline_result.data_only_ensemble,
                    data_only_history=baseline_result.data_only_history,
                    data_only_fcfw_weights=final_evaluation.data_only_fcfw_weights,
                )
                print(f"[ARTIFACTS] Circuit artifact saved to: {artifact_path}")
            except Exception as e:
                print(f"[ARTIFACTS] Failed to save circuit artifact: {e}")

        return {
            'final_stats': final_evaluation.final_stats,
            'ensemble_metrics_history': ensemble_metrics_history,
            'ensemble': ensemble,
            'baseline_stats': baseline_result.standalone_stats,
            'data_only_stats': baseline_result.data_only_stats,
            'ensemble_fcfw_stats': final_evaluation.ensemble_fcfw_stats,
            'ensemble_fcfw_weights': final_evaluation.ensemble_fcfw_weights,
            'output_dir': str(output.run_dir),
            'weights': np.asarray(ensemble.weights, dtype=np.float64),
            'n_models_accepted': len(ensemble.models),
        }
