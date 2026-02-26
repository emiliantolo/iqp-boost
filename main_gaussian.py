"""Ensemble boosting on Gaussian mixture dataset."""

import iqpopt as iqp
import iqpopt.gen_qml as gen
from iqpopt.gen_qml.utils import median_heuristic
from iqpopt.gen_qml.sample_methods import mmd_loss_samples
from src.datasets.gaussian import GaussianMixtureDataset
from src.ensemble import BoostedEnsemble
from src.reporting import (
    report_metrics_table, get_plot_config, OutputManager, plot_data_ensemble_loss,
    plot_metrics_progression, report_baseline, report_final, report_rejection, report_step
)
from src.core import (
    setup_iqp_circuit, sample_ensemble, evaluate_samples,
    apply_weight_strategy, check_acceptance, get_params_init
)
import jax
import jax.numpy as jnp
import numpy as np
import gc
import matplotlib.pyplot as plt


CONFIG = {
    'dims': (4, 4),
    'n_samples_train': 800,
    'data_seed': 42,
    'sigma_factor': 1.0,
    'n_ops': 80,
    'n_samples': 250,
    'n_models': 8,
    'learning_rate': 0.01,
    'epochs_per_step': 200,
    'cache_ensemble_traces': False,
    'ensemble_samples_coeff': 1.0,
    'lambda_dual': 1.0,
    'weight_strategy': 'greedy',
    'eval_samples': 400,
    'keep_models_for_diagnosis': True,  # Set True to keep all models regardless of MMD
    'stop_on_reject': False,  # Set True to stop on first rejection
    'gaussian_separation': 3.0,
    'gaussian_spread': 0.3,
    'init_baseline': 'random',
    'init_later': 'random',
    'compute_kl': True,
    'rng_seed': 42,
}


def pairwise_model_mmd(models: list, circuit: iqp.IqpSimulator, sigma: float, 
                       n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Compute pairwise MMD between models."""
    n_models = len(models)
    pairwise_mmd = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        samples_i = circuit.sample(models[i], shots=n_samples)
        for j in range(i+1, n_models):
            samples_j = circuit.sample(models[j], shots=n_samples)
            mmd_ij = float(mmd_loss_samples(samples_i, samples_j, sigma))
            pairwise_mmd[i, j] = mmd_ij
            pairwise_mmd[j, i] = mmd_ij
    
    return pairwise_mmd


def main():
    """Test ensemble boosting on Gaussian mixture data."""
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
    grid_size = dims[0]  # Assume square grid
    n_qubits = 8  # 4 bits per dimension (x, y)
    
    # Create Gaussian mixture dataset
    print("Generating Gaussian mixture data...")
    gaussian_dataset = GaussianMixtureDataset(
        grid_size=grid_size,
        spread=CONFIG['gaussian_spread'],
        separation=CONFIG['gaussian_separation'],
        grid_bits=4
    )
    x_train = gaussian_dataset.generate(n_samples=CONFIG['n_samples_train'], seed=CONFIG['data_seed'])
    print(f"Generated {len(x_train)} samples, {x_train.shape[1]} qubits")
    
    circuit, gates, gate_desc = setup_iqp_circuit(n_qubits, topology='random', min_weight=2, max_weight=2, n_gates=4)
    print(f"Gate structure: {gate_desc}")
    
    # Create Gaussian-specific validator functions
    def gaussian_validity(samples):
        return gaussian_dataset.validity_rate(samples, threshold_std=2.0)
    
    def gaussian_coverage(ground_truth, samples):
        return gaussian_dataset.coverage_rate(ground_truth, samples, threshold_std=2.0)
    
    # =======================================================

    
    sigma_base = median_heuristic(x_train)
    sigma = CONFIG['sigma_factor'] * sigma_base
    print(f"sigma_base={sigma_base:.4f}, sigma={sigma:.4f}")
    
    key = jax.random.PRNGKey(0)
    epochs_base = CONFIG['epochs_per_step'] * CONFIG['n_models']
    
    params_init = get_params_init(CONFIG['init_baseline'], gates, x_train, key)
    
    print(f"\nTraining baseline model ({epochs_base} epochs)...")
    print(f"Circuit: {n_qubits} qubits, {len(params_init)} parameters")
    loss_kwargs = {
        "params": params_init, "iqp_circuit": circuit, "ground_truth": x_train,
        "sigma": sigma, "n_ops": CONFIG['n_ops'], "n_samples": CONFIG['n_samples'],
    }
    trainer_baseline = iqp.Trainer("Adam", gen.mmd_loss_iqp, stepsize=CONFIG['learning_rate'])
    monitor_interval = get_plot_config()['plot_interval'] if get_plot_config()['plot_data_loss'] else None
    trainer_baseline.train(n_iters=epochs_base, loss_kwargs=loss_kwargs, monitor_interval=50, turbo=10)
    
    base_losses = getattr(trainer_baseline, "losses", [])
    if len(base_losses) > 0:
        print(f"Baseline training: initial loss={float(base_losses[0]):.6f}, final loss={float(base_losses[-1]):.6f}")
    else:
        print("Warning: No loss history recorded for baseline!")
    
    eval_samples = CONFIG['eval_samples']
    rng = np.random.default_rng(0)
    baseline_samples = circuit.sample(trainer_baseline.final_params, shots=eval_samples)
    baseline_stats = evaluate_samples(x_train, baseline_samples, sigma, gaussian_validity, gaussian_coverage,
                                     compute_kl=CONFIG['compute_kl'])
    
    print()
    report_baseline(baseline_stats['mmd'], baseline_stats)
    print(f"  KL={baseline_stats['kl']:.6f}, F1={100*baseline_stats['f_score']:.1f}%, "
        f"Coverage={100*baseline_stats['coverage']:.1f}%")

    # Train ensemble
    ensemble = BoostedEnsemble(
        circuit, n_models=CONFIG['n_models'], sigma=sigma,
        n_ops=CONFIG['n_ops'], n_samples=CONFIG['n_samples'],
        cache_ensemble_traces=CONFIG['cache_ensemble_traces'],
        ensemble_samples_coeff=CONFIG['ensemble_samples_coeff'],
        lambda_dual=CONFIG['lambda_dual'],
    )
    
    print("\nTraining ensemble...")
    print(f"[Step 0] Training base model ({CONFIG['epochs_per_step']} epochs)...")
    
    eval_key = jax.random.PRNGKey(123)
    key = ensemble.train_base(ground_truth=x_train, key=key, steps=CONFIG['epochs_per_step'],
                              stepsize=CONFIG['learning_rate'], monitor_interval=50,
                              init_strategy=CONFIG['init_baseline'], turbo=10)
    
    ens_samples = sample_ensemble(ensemble, circuit, eval_samples, rng)
    ens_stats = evaluate_samples(x_train, ens_samples, sigma, gaussian_validity, gaussian_coverage,
                                compute_kl=CONFIG['compute_kl'])
    
    # Track ensemble MMD at each step
    ensemble_metrics_history = {
        'mmd': [ens_stats['mmd']],
        'validity': [ens_stats['validity']],
        'coverage': [ens_stats['coverage']],
        'kl': [ens_stats['kl']],
        'f_score': [ens_stats['f_score']],
    }
    # Track accepted steps
    accepted_steps = [0]  # Step 0 (base model) always accepted
    all_mmd_values = [ens_stats['mmd']]  # Track all MMD values (accepted and rejected)
    all_step_positions = []  # Track step numbers for all trained models
    
    report_step(0, CONFIG['n_models'], ens_stats['mmd'])
    print(f"  F1={100*ens_stats['f_score']:.1f}%, Coverage={100*ens_stats['coverage']:.1f}%")
    
    prev_ens_stats = ens_stats
    
    # Boosting loop
    for step in range(1, CONFIG['n_models']):
        print(f"\n[Step {step}] Training new model...")
        snapshot = ensemble.snapshot_state()
        
        # All boosted models use init_later
        key = ensemble.step(ground_truth=x_train, key=key, steps=CONFIG['epochs_per_step'],
                           stepsize=CONFIG['learning_rate'], verbose=False,
                           monitor_interval=monitor_interval, init_strategy=CONFIG['init_later'])
        
        # Optimize weights
        weight_strategy = CONFIG.get('weight_strategy', 'greedy')
        apply_weight_strategy(ensemble, snapshot, rng, circuit, x_train, sigma, eval_samples, weight_strategy)
        
        # Evaluate ensemble
        ens_samples = sample_ensemble(ensemble, circuit, eval_samples, rng)
        ens_stats = evaluate_samples(x_train, ens_samples, sigma, gaussian_validity, gaussian_coverage,
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
        ensemble_metrics_history['mmd'].append(ens_stats['mmd'])
        ensemble_metrics_history['validity'].append(ens_stats['validity'])
        ensemble_metrics_history['coverage'].append(ens_stats['coverage'])
        ensemble_metrics_history['kl'].append(ens_stats['kl'])
        ensemble_metrics_history['f_score'].append(ens_stats['f_score'])
        report_step(step, CONFIG['n_models'], ens_stats['mmd'])
        print(f"  Models: {len(ensemble.models)}, F1={100*ens_stats['f_score']:.1f}%, "
            f"Coverage={100*ens_stats['coverage']:.1f}%")
    
        prev_ens_stats = ens_stats
        gc.collect()
        if step % 2 == 0:
            jax.clear_caches()
    
    report_final(baseline_stats['mmd'], ens_stats['mmd'], len(ensemble.models), ens_stats)
    print(f"  F1={ens_stats['f_score']:.4f}, Coverage={ens_stats['coverage']:.4f}, KL={ens_stats['kl']:.4f}")

    # Final per-model metrics table (baseline + ensemble + each model)
    model_samples_list = []
    model_rows = []
    for i, model_params in enumerate(ensemble.models):
        model_samples = circuit.sample(model_params, shots=eval_samples)
        model_samples_list.append(model_samples)
        model_stats = evaluate_samples(x_train, model_samples, sigma, gaussian_validity, gaussian_coverage,
                                      compute_kl=CONFIG['compute_kl'])
        model_rows.append((f"Model {i}", model_stats))
    
    report_metrics_table(baseline_stats, ens_stats, model_rows, "FINAL MODEL COMPARISON")

    if get_plot_config()['plot_data_loss']:
        # Pass baseline losses as-is (epochs 0 to len-1)
        plot_data_ensemble_loss(ensemble, ensemble_metrics_history, baseline_stats, output,
                                baseline_train_losses=base_losses, accepted_steps=accepted_steps,
                                all_mmd_values=all_mmd_values, all_step_positions=all_step_positions)
        
        # Plot metrics progression (for Gaussian, use f_score instead of validity)
        gaussian_metric_configs = [
            ('mmd', 'MMD²', 1, 'blue', 'o'),
            ('f_score', 'F1 Score (%)', 100, 'green', 's'),
            ('coverage', 'Coverage (%)', 100, 'purple', '^'),
            ('kl', 'KL(data || model)', 1, 'orange', 'd'),
        ]
        plot_metrics_progression(ensemble_metrics_history, baseline_stats, output, gaussian_metric_configs)

    # Collect samples for visualization
    baseline_model_samples = baseline_samples

    # Visualization: Scatter plot with baseline + each model
    print(f"\n  Generating visualization of baseline and models...")
    try:
        # Decode to 2D
        x_train_2d_recovered = gaussian_dataset.binary_to_continuous(x_train)
        baseline_2d = gaussian_dataset.binary_to_continuous(baseline_model_samples)
        
        fig, ax_scatter = plt.subplots(figsize=(12, 9))
        
        # Plot ground truth data as crosses
        ax_scatter.scatter(x_train_2d_recovered[:, 0], x_train_2d_recovered[:, 1], 
                   alpha=0.5, s=100, c='black', marker='x', linewidths=2.0,
                   label='Ground Truth Data', zorder=2)
        
        # Plot baseline as large star
        baseline_mean_x = np.mean(baseline_2d[:, 0])
        baseline_mean_y = np.mean(baseline_2d[:, 1])
        baseline_std_x = np.std(baseline_2d[:, 0])
        baseline_std_y = np.std(baseline_2d[:, 1])
        
        ax_scatter.errorbar(baseline_mean_x, baseline_mean_y, xerr=baseline_std_x, yerr=baseline_std_y,
                   color='purple', alpha=0.6, linewidth=2.5, capsize=6, capthick=2.0, zorder=3)
        ax_scatter.scatter(baseline_mean_x, baseline_mean_y, s=500, c='purple', marker='*',
                  edgecolors='darkviolet', linewidth=2.0, label='Baseline (mean+/-std)', zorder=4)
        
        # Plot each model: mean point with std error bars
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_samples_list)))
        for i, model_samples in enumerate(model_samples_list):
            model_2d = gaussian_dataset.binary_to_continuous(model_samples)
            mean_x = np.mean(model_2d[:, 0])
            mean_y = np.mean(model_2d[:, 1])
            std_x = np.std(model_2d[:, 0])
            std_y = np.std(model_2d[:, 1])
            
            ax_scatter.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y,
                       color=colors[i], alpha=0.5, linewidth=2.0,
                       capsize=5, capthick=2.0, zorder=3)
            ax_scatter.scatter(mean_x, mean_y, s=250, c=[colors[i]], marker='o',
                      edgecolors='darkgray', linewidth=1.5, label=f'Model {i} (mean+/-std)',
                      zorder=4)
        
        ax_scatter.set_xlabel('X', fontsize=12, fontweight='bold')
        ax_scatter.set_ylabel('Y', fontsize=12, fontweight='bold')
        ax_scatter.set_title('Ground Truth vs Baseline + Individual Models', fontsize=13, fontweight='bold')
        ax_scatter.legend(loc='best', fontsize=9, ncol=1)
        ax_scatter.grid(True, alpha=0.3)
        # Auto-scale plot limits based on data range (handles both [0,1] and [-0.89, 9.8] scales)
        ax_scatter.margins(0.1)

        plt.tight_layout()
        plot_path = output.get_path('ensemble_model_distributions.pdf')
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f"  [OK] Saved plot to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"  [ERROR] Could not generate plot: {e}")
    
    print("\n" + "="*80)
    
    # Close output manager
    output.__exit__(None, None, None)


if __name__ == "__main__":
    main()
