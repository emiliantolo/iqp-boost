"""Ensemble boosting on BAS (Bars and Stripes) dataset."""

import iqpopt as iqp
import iqpopt.gen_qml as gen
from iqpopt.gen_qml.utils import median_heuristic
from src.datasets.blobs import BlobsDataset
from src.ensemble import BoostedEnsemble
from src.reporting import (
    report_metrics_table, get_plot_config, OutputManager, plot_data_ensemble_loss,
    plot_metrics_progression, report_baseline, report_final, report_rejection,
    report_step, report_gradient_snr
)
from src.core import (
    setup_iqp_circuit, sample_ensemble, evaluate_samples,
    apply_weight_strategy, check_acceptance, get_params_init
)
from src.dual_mmd_loss import gradient_snr
import jax
import jax.numpy as jnp
import numpy as np
import gc
import matplotlib.pyplot as plt


CONFIG = {
    'dims': (4, 4),
    'n_samples_train': 1000,
    'sigma_factor': 0.5,
    'n_ops': 1000,
    'n_samples': 2000,
    'n_models': 8,
    'learning_rate': 0.005,
    'epochs_per_step': 800,
    'cache_ensemble_traces': False,
    'ensemble_samples_coeff': 1.0,
    'init_baseline': 'covariance',
    'init_later': 'random',
    'lambda_dual': 1.0,
    'weight_strategy': 'line_search',
    'eval_samples': 1000,
    'keep_models_for_diagnosis': True,  # Set True to keep all models regardless of MMD
    'stop_on_reject': False,  # Set True to stop on first rejection
    'compute_kl': True,
    'rng_seed': 42,
}

def main():
    # Set random seeds for reproducibility
    np.random.seed(CONFIG['rng_seed'])
    
    # Set up output directory and logging
    output = OutputManager(base_dir='out', run_name=None)
    output.__enter__()
    
    # Log configuration parameters
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    dims = CONFIG['dims']
    n_qubits = dims[0] * dims[1]
    
    # Create Blobs dataset
    blobs_dataset = BlobsDataset()
    x_train = blobs_dataset.generate(n_samples=CONFIG['n_samples_train'])
    
    # Create Blobs-specific validator functions
    def blobs_validity(samples):
        return blobs_dataset.validity_rate(samples)
    
    def blobs_coverage(ground_truth, samples):
        return blobs_dataset.coverage_rate(ground_truth, samples)
    
    circuit, gates, gate_desc = setup_iqp_circuit(n_qubits, topology='neighbour', distance=3, max_weight=2)
    print(f"Gate structure: {gate_desc}")

    sigma_base = median_heuristic(x_train)
    sigma = CONFIG['sigma_factor'] * sigma_base
    print(f"sigma_base={sigma_base:.4f}, sigma={sigma:.4f}")

    key = jax.random.PRNGKey(CONFIG['rng_seed'])
    epochs_base = CONFIG['epochs_per_step'] * CONFIG['n_models']

    params_init = get_params_init(CONFIG['init_baseline'], gates, x_train, key)
    loss_kwargs = {"params": params_init, "iqp_circuit": circuit, "ground_truth": x_train,
                   "sigma": sigma, "n_ops": CONFIG['n_ops'], "n_samples": CONFIG['n_samples']}
    trainer_baseline = iqp.Trainer("Adam", gen.mmd_loss_iqp, stepsize=CONFIG['learning_rate'])
    plot_cfg = get_plot_config()
    monitor_interval = plot_cfg['plot_interval'] if plot_cfg['plot_data_loss'] else None
    trainer_baseline.train(n_iters=epochs_base, loss_kwargs=loss_kwargs, monitor_interval=monitor_interval, turbo=10)
    base_losses = getattr(trainer_baseline, "losses", [])
    if len(base_losses) > 0:
        print(f"Baseline final loss: {float(base_losses[-1]):.6f}")
    eval_samples = CONFIG['eval_samples']
    rng = np.random.default_rng(0)
    baseline_samples = circuit.sample(trainer_baseline.final_params, shots=eval_samples)
    baseline_stats = evaluate_samples(x_train, baseline_samples, sigma, blobs_validity, blobs_coverage,
                                     compute_kl=CONFIG['compute_kl'])
    print()
    report_baseline(baseline_stats['mmd'], baseline_stats)
    print(f"  KL={baseline_stats['kl']:.6f}, Validity={baseline_stats['validity']:.4f}, "
          f"Coverage={baseline_stats['coverage']:.4f}")

    ensemble = BoostedEnsemble(circuit, n_models=CONFIG['n_models'], sigma=sigma, 
        n_ops=CONFIG['n_ops'], n_samples=CONFIG['n_samples'],
        cache_ensemble_traces=CONFIG['cache_ensemble_traces'], 
        ensemble_samples_coeff=CONFIG['ensemble_samples_coeff'],
        lambda_dual=CONFIG['lambda_dual'],
    )
    print("Training ensemble...")
    eval_key = jax.random.PRNGKey(123)

    key = ensemble.train_base(ground_truth=x_train, key=key, steps=CONFIG['epochs_per_step'],
                              stepsize=CONFIG['learning_rate'], monitor_interval=monitor_interval,
                              init_strategy=CONFIG['init_baseline'], turbo=10)
    ens_samples = sample_ensemble(ensemble, circuit, eval_samples, rng)
    ens_stats = evaluate_samples(x_train, ens_samples, sigma, blobs_validity, blobs_coverage,
                                compute_kl=CONFIG['compute_kl'])
    
    report_step(0, CONFIG['n_models'], ens_stats['mmd'])
    print(f"  KL={ens_stats['kl']:.6f}, Validity={ens_stats['validity']:.4f}, "
          f"Coverage={ens_stats['coverage']:.4f}")
    
    prev_ens_stats = ens_stats

    # Track ensemble MMD at each step
    ensemble_mmd_history = [baseline_stats['mmd']]
    # Track all metrics at each step
    ensemble_metrics_history = {
        'mmd': [ens_stats['mmd']],
        'validity': [ens_stats['validity']],
        'coverage': [ens_stats['coverage']],
        'kl': [ens_stats['kl']],
    }
    # Track SNR history and accepted steps
    snr_history = []
    accepted_steps = [0]  # Step 0 (base model) always accepted
    all_mmd_values = [ens_stats['mmd']]  # Track all MMD values (accepted and rejected)
    all_step_positions = []  # Track epoch positions for all steps

    for step in range(1, CONFIG['n_models']):
        print(f"\n[Step {step}] Training new model...")
        snapshot = ensemble.snapshot_state()
        
        # All boosted models use init_later
        key = ensemble.step(ground_truth=x_train, key=key, steps=CONFIG['epochs_per_step'],
                stepsize=CONFIG['learning_rate'], verbose=True,
                monitor_interval=monitor_interval, init_strategy=CONFIG['init_later'])
        
        # Optimize weights
        weight_strategy = CONFIG.get('weight_strategy', 'greedy')
        apply_weight_strategy(ensemble, snapshot, rng, circuit, x_train, sigma, eval_samples, weight_strategy)
        
        # Evaluate ensemble
        ens_samples = sample_ensemble(ensemble, circuit, eval_samples, rng)
        ens_stats = evaluate_samples(x_train, ens_samples, sigma, blobs_validity, blobs_coverage,
                                compute_kl=CONFIG['compute_kl'])
        
        # Check acceptance
        accepted, should_stop = check_acceptance(
            ens_stats['mmd'], prev_ens_stats['mmd'], ensemble, snapshot, step,
            keep_all=CONFIG.get('keep_models_for_diagnosis', False),
            stop_on_reject=CONFIG.get('stop_on_reject', False)
        )
        
        if should_stop:
            break
        
        # Track this step's final MMD (whether accepted or rejected)
        all_mmd_values.append(ens_stats['mmd'])
        all_step_positions.append(step)
        
        if not accepted:
            report_rejection(step)
            continue

        # Track accepted step
        accepted_steps.append(step)

        # Track progress
        ensemble_mmd_history.append(ens_stats['mmd'])
        ensemble_metrics_history['mmd'].append(ens_stats['mmd'])
        ensemble_metrics_history['validity'].append(ens_stats['validity'])
        ensemble_metrics_history['coverage'].append(ens_stats['coverage'])
        ensemble_metrics_history['kl'].append(ens_stats['kl'])
        report_step(step, CONFIG['n_models'], ens_stats['mmd'])
        print(f"  KL={ens_stats['kl']:.6f}, Validity={ens_stats['validity']:.4f}, "
              f"Coverage={ens_stats['coverage']:.4f}")
        
        # Compute SNR of the newly added model (debug only, not plotted)
        try:
            new_params = ensemble.models[-1]
            # Wrapper: gradient_snr will bind iqp_circuit, x_train, and loss_kwargs via functools.partial
            def mmd_loss_for_snr(params, iqp_circuit, x_train, key, sigma, n_ops, mmd_n_samples):
                return gen.mmd_loss_iqp(params, iqp_circuit, x_train, sigma, n_ops, mmd_n_samples, key)
            # Use default n_samples=10 for gradient_snr (number of gradient samples for SNR)
            # Pass mmd_n_samples separately to avoid parameter name collision
            snr_info = gradient_snr(
                new_params, circuit, x_train,
                mmd_loss_for_snr, key,
                sigma=sigma, n_ops=CONFIG['n_ops'], mmd_n_samples=CONFIG['n_samples']
            )
            snr_history.append(snr_info['snr'])
            report_gradient_snr(snr_info)
        except Exception as e:
            print(f"  [Debug] Gradient SNR computation failed: {str(e)[:80]}")
            snr_history.append(np.nan)
        
        prev_ens_stats = ens_stats
        gc.collect()
        if step % 2 == 0:
            jax.clear_caches()

    final_ensemble_samples = sample_ensemble(ensemble, circuit, eval_samples, rng)
    final_stats = evaluate_samples(x_train, final_ensemble_samples, sigma, blobs_validity, blobs_coverage,
                                  compute_kl=CONFIG['compute_kl'])
    
    report_final(baseline_stats['mmd'], final_stats['mmd'], len(ensemble.models), final_stats)
    print(f"  KL={final_stats['kl']:.6f}, Validity={final_stats['validity']:.4f}, "
          f"Coverage={final_stats['coverage']:.4f}")
    
    # Final per-model metrics table
    model_rows = []
    for i, model_params in enumerate(ensemble.models):
        model_samples = circuit.sample(model_params, shots=eval_samples)
        model_stats = evaluate_samples(x_train, model_samples, sigma, blobs_validity, blobs_coverage,
                                      compute_kl=CONFIG['compute_kl'])
        model_rows.append((f"Model {i}", model_stats))
    
    report_metrics_table(baseline_stats, final_stats, model_rows, "FINAL MODEL COMPARISON")

    if get_plot_config()['plot_data_loss']:
        # Pass baseline losses as-is (epochs 0 to len-1)
        plot_data_ensemble_loss(ensemble, ensemble_metrics_history, baseline_stats, output, 
                                baseline_train_losses=base_losses, accepted_steps=accepted_steps,
                                all_mmd_values=all_mmd_values, all_step_positions=all_step_positions)
        
        # Plot metrics progression (for BAS, use validity instead of f_score)
        bas_metric_configs = [
            ('mmd', 'MMD²', 1, 'blue', 'o'),
            ('validity', 'Validity (%)', 100, 'green', 's'),
            ('coverage', 'Coverage (%)', 100, 'purple', '^'),
            ('kl', 'KL(data || model)', 1, 'orange', 'd'),
        ]
        plot_metrics_progression(ensemble_metrics_history, baseline_stats, output, bas_metric_configs)
        
    # Visualization of sample shapes from each model
    print(f"\nGenerating blobs visualizations...")
    try:
        n_models_to_show = min(3, len(ensemble.models))
        fig, axes = plt.subplots(n_models_to_show, 4, figsize=(14, 4*n_models_to_show))
        if n_models_to_show == 1:
            axes = axes.reshape(1, -1)
        
        for model_idx in range(n_models_to_show):
            # Sample from this model
            model_samples = circuit.sample(ensemble.models[model_idx], shots=eval_samples)
            
            # Show 4 random samples
            indices = np.random.choice(len(model_samples), size=4, replace=False)
            for i, sample_idx in enumerate(indices):
                blobs_dataset.visualize(model_samples[sample_idx], ax=axes[model_idx, i])
                axes[model_idx, i].set_title(f"Model {model_idx}\nSample {sample_idx}")
        
        plt.tight_layout()
        plot_path = output.get_path("blobs_samples.png")
        plt.savefig(plot_path, format='png', bbox_inches='tight', dpi=100)
        print(f"Saved blobs samples visualization to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Could not save blobs visualization: {e}")
        
    # Close output manager
    output.__exit__(None, None, None)


if __name__ == "__main__":
    main()
