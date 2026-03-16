import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pennylane as qml

# Plot configuration
PLOT_CONFIG = {
    'plot_data_loss': True,
    'plot_interval': 10,
}


def get_plot_config():
    """Get plot configuration."""
    return PLOT_CONFIG.copy()


def set_plot_config(config_dict: dict):
    """Update plot configuration."""
    PLOT_CONFIG.update(config_dict)

def report_step(step: int, n_models: int,
                training_mmd: float = None, sampled_mmd: float = None,
                data_loss: float = None, ens_loss: float = None):
    """Report single boosting step result."""
    parts = [f"[Step {step}/{n_models-1}]"]
    if training_mmd is not None:
        parts.append(f"train_MMD={training_mmd:.4f}")
    if sampled_mmd is not None:
        parts.append(f"sample_MMD={sampled_mmd:.4f}")
    if data_loss is not None:
        parts.append(f"data={data_loss:.4f}")
    if ens_loss is not None:
        parts.append(f"ens={ens_loss:.4f}")
    print("  " + " ".join(parts))


def report_baseline(baseline_mmd: float, baseline_stats: dict = None):
    """Report baseline model metrics."""
    print(f"Baseline: MMD={baseline_mmd:.4f}", end="")
    if baseline_stats and 'f_score' in baseline_stats:
        print(f" F1={baseline_stats['f_score']:.3f}", end="")
    print()


def report_final(baseline_mmd: float, final_mmd: float, n_models: int, final_stats: dict = None):
    """Report final ensemble metrics."""
    improvement = baseline_mmd - final_mmd
    print(f"\nFinal: {n_models} models, MMD={final_mmd:.4f} (delta={improvement:+.4f})", end="")
    if final_stats and 'f_score' in final_stats:
        print(f" F1={final_stats['f_score']:.3f}", end="")
    print()


def report_rejection(step: int):
    """Report model rejection."""
    print(f"  [Step {step}] REJECTED (no improvement)")


def report_gradient_snr(snr_info: dict):
    """Report gradient SNR metrics."""
    if snr_info:
        print(f"  SNR (proxy): {snr_info['snr']:.2f} " +
              f"(mean_norm={snr_info['mean_grad_norm']:.4f}, " +
              f"std_norm={snr_info['std_grad_norm']:.4f})")


def report_config(config: dict, dataset_name: str):
    """Report experiment configuration."""
    print(f"Experiment: Boosting on {dataset_name}")
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*80)


def report_circuit(gate_desc: str):
    """Report circuit structure."""
    print(f"Gate structure: {gate_desc}")


def save_circuit_plot(circuit, output_manager, filename: str = 'circuit_structure.pdf'):
    """Save a PennyLane-rendered plot of the IQP circuit."""
    try:
        init_coefs = np.zeros(len(circuit.init_gates)) if circuit.init_gates else None
        fig, _ = qml.draw_mpl(
            lambda params: circuit.iqp_circuit(params, init_coefs)
        )(jnp.zeros(circuit.n_gates))
        plot_path = output_manager.get_path(filename)
        fig.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=150)
        print(f"Saved circuit structure plot to: {plot_path}")
    except Exception as e:
        print(f"Failed to save circuit plot: {e}")

def report_kernel(sigma: float | list, n_ops: int, n_qubits: int):
    """Report kernel configuration."""
    sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
    print(f"Using Gaussian Kernel (n_ops={n_ops}, n_qubits={n_qubits}):")
    for s in sigmas:
        print(f"  - sigma={s:.8f}")
    ps = np.array([(1 - np.exp(-1 / 2 / sigma**2)) / 2 for sigma in sigmas])
    var_within = n_qubits * (ps * (1 - ps)).mean()
    var_between = (n_qubits ** 2) * np.var(ps)
    print("Expected sampled operator size:")
    for p in ps:
        print(f"  - p={n_qubits * p:.6f}, std={np.sqrt(n_qubits * p * (1 - p)):.6f}")
    print(f"  mean: {n_qubits * ps.mean():.6f}")
    print(f"  std: {np.sqrt(var_within + var_between):.6f}")


def report_acceptance(accepted: bool, delta_mmd: float, should_stop: bool = False, diagnostic_mode: bool = False):
    """Report model acceptance or rejection with details."""
    if diagnostic_mode:
        if delta_mmd <= 0:
            print(f"  [WARNING] MMD worsened but kept (diagnostic, delta={delta_mmd:+.6f})")
        else:
            print(f"  [OK] MMD improved (delta={delta_mmd:+.6f})")
        return

    if accepted:
        print(f"  [ACCEPT] MMD improved (delta={delta_mmd:+.6f})")
    else:
        print(f"  [REJECT] MMD did not improve (delta={delta_mmd:+.6f})")
        if should_stop:
            print(f"  Stopping boosting (stop_on_reject=True)")


def report_loss_components(data_loss: float, ens_loss: float, dual: float, epoch_range: tuple = None):
    """Report loss components."""
    prefix = ""
    if epoch_range:
        prefix = f"  Captured evaluations (epochs {epoch_range[0]} to {epoch_range[1]}): "
    print(f"{prefix}Final: data_loss={data_loss:.6f}, ens_loss={ens_loss:.6f}, dual={dual:.6f}")


def report_operators(label: str, n_ops: int, n_qubits: int, avg_term_size: float, std_term_size: float, 
                     min_size: int, max_size: int, sigma: float):
    """Report operator statistics."""
    print(f"[{label}] n_ops={n_ops}, n_qubits={n_qubits}, avg_term_size={avg_term_size:.2f}, "
          f"std_term_size={std_term_size:.2f}, min={min_size}, max={max_size}, sigma={sigma}")


def report_metrics_table(baseline_stats: dict, ensemble_stats: dict, model_list: list = None, title: str = "FINAL METRICS"):
    """
    Report unified metrics table with all metrics (MMD, KL, validity, coverage, precision, recall, F1).
    
    Args:
        baseline_stats: Dict with baseline metrics
        ensemble_stats: Dict with ensemble metrics
        model_list: List of (name, stats_dict) tuples for individual models (optional)
        title: Table title
    """
    def _fmt(stats: dict, key: str, mode: str) -> str:
        if key not in stats or stats[key] is None:
            return "n/a"
        value = stats[key]
        if mode == 'float4':
            return f"{value:.4f}"
        if mode == 'float3':
            return f"{value:.3f}"
        if mode == 'pct1':
            return f"{100*value:.1f}%"
        return str(value)

    # Show concise, decision-relevant metrics
    headers = ["Model", "MMD", "TVD", "Valid", "Cover", "F1"]
    rows = [
        ["Baseline", 
         _fmt(baseline_stats, 'mmd', 'float4'),
         _fmt(baseline_stats, 'tvd', 'float3'),
         _fmt(baseline_stats, 'validity', 'pct1'),
         _fmt(baseline_stats, 'coverage', 'pct1'),
         _fmt(baseline_stats, 'f_score', 'pct1')],
        ["Ensemble", 
         _fmt(ensemble_stats, 'mmd', 'float4'),
         _fmt(ensemble_stats, 'tvd', 'float3'),
         _fmt(ensemble_stats, 'validity', 'pct1'),
         _fmt(ensemble_stats, 'coverage', 'pct1'),
         _fmt(ensemble_stats, 'f_score', 'pct1')],
    ]
    if model_list:
        for name, stats in model_list:
            rows.append([name,
                        _fmt(stats, 'mmd', 'float4'),
                        _fmt(stats, 'tvd', 'float3'),
                        _fmt(stats, 'validity', 'pct1'),
                        _fmt(stats, 'coverage', 'pct1'),
                        _fmt(stats, 'f_score', 'pct1')])
    
    print(f"\n{title}")
    header_line = " | ".join(f"{h:<10}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        row_line = " | ".join(f"{str(v):<10}" for v in row)
        print(row_line)


def plot_data_ensemble_loss(ensemble, ensemble_metrics_history, baseline_stats, output_manager,
                           baseline_train_losses=None, data_only_history=None):
    """
    Plot training and sampled ensemble MMD progression plus loss history.
    
    This function creates a 2-panel plot:
    - Left: Ensemble MMD wrt data per accepted step (sampled + analytical training estimate)
    - Right: Dual loss components across training epochs with step separators
    
    Args:
        ensemble: BoostedEnsemble object with training_losses attribute
        ensemble_metrics_history: Dict with accepted-step metric history
        baseline_stats: Baseline metrics dict
        output_manager: OutputManager instance for saving plots
        baseline_train_losses: Optional array of baseline training losses (if trained separately)
        data_only_history: Optional metrics history dict for data-only iterative baseline
    """
    if not hasattr(ensemble, 'training_losses') or len(ensemble.training_losses) == 0:
        print("No training losses available for plotting")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Left panel: baseline + model 0 loss history + ensemble/data-only final step losses
    ax = axes[0]
    steps = np.array(ensemble_metrics_history.get('step', np.arange(len(ensemble_metrics_history.get('training_loss', [])))))
    train_mmd = np.array(ensemble_metrics_history.get('training_loss', []), dtype=float)

    # Baseline training curve (gray) on its own epoch axis
    if baseline_train_losses is not None and len(baseline_train_losses) > 0:
        baseline_x = np.arange(len(baseline_train_losses))
        ax.plot(baseline_x, baseline_train_losses, '-', linewidth=1.8, color='gray',
                alpha=0.8, label='Baseline loss curve', zorder=1)

    # Model 0 training curve starts at epoch 0
    model0_final = None
    model0_epoch_end = 0
    if len(ensemble.training_losses) > 0 and 'total' in ensemble.training_losses[0]:
        m0_losses = ensemble.training_losses[0]['total']
        if len(m0_losses) > 0:
            m0_x = np.arange(len(m0_losses))
            model0_final = float(m0_losses[-1])
            model0_epoch_end = m0_x[-1] + 1
            ax.plot(m0_x, m0_losses, '-', linewidth=2.0, color='blue',
                    alpha=0.9, label='Model 0 loss curve', zorder=3)
            ax.plot([model0_epoch_end - 1], [model0_final], 'o', color='blue', markersize=8, zorder=5)

    # Ensemble data loss points at cumulative end-epochs of accepted models
    if len(train_mmd) > 0:
        ens_y = np.array(train_mmd, dtype=float)

        # Build x positions from actual training history: end epoch of each accepted model
        accepted_epoch_positions = []
        cumulative = 0
        for loss_dict in ensemble.training_losses:
            total_loss = loss_dict.get('total', [])
            if len(total_loss) == 0:
                continue
            cumulative += len(total_loss)
            accepted_epoch_positions.append(cumulative - 1)

        if len(accepted_epoch_positions) >= len(ens_y):
            ensemble_steps = np.array(accepted_epoch_positions[:len(ens_y)], dtype=float)
        else:
            # Fallback: approximate with constant step span inferred from boosted model lengths
            boosted_lens = [len(ld.get('total', [])) for ld in ensemble.training_losses[1:] if len(ld.get('total', [])) > 0]
            step_span = boosted_lens[0] if boosted_lens else max(model0_epoch_end, 1)
            ensemble_steps = (model0_epoch_end - 1) + steps * step_span

        if model0_final is not None and len(ens_y) > 0:
            ens_y[0] = model0_final
        ax.plot(ensemble_steps, ens_y, 'o-', markersize=7,
                color='blue', linewidth=2.0, label='Ensemble data loss', zorder=6)

    # Optional data-only trajectory starting from epoch 0
    if data_only_history is not None and 'training_loss' in data_only_history:
        data_only_steps = np.array(data_only_history.get('step', np.arange(len(data_only_history['training_loss']))))
        data_only_mmd = np.array(data_only_history.get('training_loss', []), dtype=float)
        if len(data_only_mmd) > 0:
            boosted_lens = [len(ld.get('total', [])) for ld in ensemble.training_losses[1:] if len(ld.get('total', [])) > 0]
            step_span = boosted_lens[0] if boosted_lens else max(model0_epoch_end, 1)
            data_only_epoch_steps = data_only_steps * step_span
            ax.plot(data_only_epoch_steps, data_only_mmd, 'o--', markersize=5,
                    color='tab:orange', linewidth=1.8, label='Data-only loss', zorder=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMD^2")
    ax.set_title("Ensemble MMD^2 vs Data")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc='best')
    
    # Right panel: Dual loss curves per model with final data/ensemble components
    ax = axes[1]
    if len(ensemble.training_losses) > 1:
        # Start from end of model_0 training (cumulative epoch position)
        model_0_epochs = len(ensemble.training_losses[0]['total'])
        cumulative_epochs = model_0_epochs
        step_boundaries = [model_0_epochs]
        
        # Color palette for models
        n_boosted = len(ensemble.training_losses) - 1
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_boosted))
        
        for i in range(1, len(ensemble.training_losses)):
            loss_dict = ensemble.training_losses[i]
            total_loss = loss_dict['total']
            epochs_full = np.arange(len(total_loss)) + cumulative_epochs
            color = colors[i - 1]
            
            # Full-resolution dual loss curve
            ax.plot(epochs_full, total_loss, '-',
                    linewidth=2, alpha=0.8, color=color, zorder=5,
                    label=f'M{i} dual' if i < 5 or i == n_boosted else None)
            
            # Component history lines (data attraction, ensemble repulsion)
            if loss_dict.get('data_hist') is not None and loss_dict.get('hist_epochs') is not None:
                hist_epochs = cumulative_epochs + loss_dict['hist_epochs']
                ax.plot(hist_epochs, loss_dict['data_hist'], '--', linewidth=1.5,
                        color=color, alpha=0.9, zorder=6, label=f'M{i} data' if i < 3 else None)
            
            if loss_dict.get('ens_hist') is not None and loss_dict.get('hist_epochs') is not None:
                hist_epochs = cumulative_epochs + loss_dict['hist_epochs']
                ax.plot(hist_epochs, -np.abs(loss_dict['ens_hist']), ':', linewidth=1.5,
                        color=color, alpha=0.9, zorder=6, label=f'M{i} ens' if i < 3 else None)
            
            cumulative_epochs += len(total_loss)
            step_boundaries.append(cumulative_epochs)
        
        # Add a zero line for reference
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Step boundaries
        for boundary in step_boundaries[1:-1]:
            ax.axvline(x=boundary, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        
        ax.set_xlabel("Epoch (cumulative)")
        ax.set_ylabel("Loss")
        ax.set_title("Dual MMD Losses (▲ data, ▼ ensemble)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc='best', fontsize='small')
    
    plt.tight_layout()
    plot_path = output_manager.get_path('data_ensemble_loss.pdf')
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Saved data/ensemble loss plot to: {plot_path}")
    plt.close()


def plot_metrics_progression(ensemble_metrics_history, baseline_stats, output_manager,
                            metric_configs=None):
    """
    Plot metrics progression across ensemble boosting steps.
    
    Args:
        ensemble_metrics_history: Dict with metric names as keys and lists of values as values
        baseline_stats: Dict with baseline metrics for reference lines
        output_manager: OutputManager instance for saving plots
        metric_configs: List of (metric_name, ylabel, scale_factor, color, marker) tuples
                       If None, uses default configuration
    
    Default config: [(mmd, ylabel, scale, color, marker), ...]
    For percentages: scale_factor = 100
    For raw values: scale_factor = 1
    """
    if metric_configs is None:
        # Default configuration for 2x2 grid: MMD, Validity/F1, Coverage, KL
        metric_configs = [
            ('mmd', 'MMD^2', 1, 'blue', 'o'),
            ('validity', 'Validity (%)', 100, 'green', 's'),
            ('coverage', 'Coverage (%)', 100, 'purple', '^'),
            ('kl', 'KL(data || model)', 1, 'orange', 'd'),
        ]
    
    # Create 2x2 subplot grid (or adapt if different number of metrics)
    n_metrics = len(metric_configs)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    
    # Flatten axes for easier indexing
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    if 'step' in ensemble_metrics_history:
        steps = np.array(ensemble_metrics_history['step'])
    else:
        steps = np.arange(len(list(ensemble_metrics_history.values())[0]))
    
    for idx, (metric_name, ylabel, scale_factor, color, marker) in enumerate(metric_configs):
        ax = axes[idx]
        
        # Check if metric is available in history
        if metric_name not in ensemble_metrics_history:
            ax.text(0.5, 0.5, f"Metric '{metric_name}' not available",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        metric_values = np.array(ensemble_metrics_history[metric_name]) * scale_factor
        baseline_value = baseline_stats.get(metric_name, 0) * scale_factor
        
        # Determine title based on metric
        if metric_name == 'validity':
            title_text = "Valid Pattern Generation"
        elif metric_name == 'f_score':
            title_text = "F1 Score"
        elif metric_name == 'coverage':
            title_text = "Ground Truth Coverage"
        elif metric_name == 'mmd':
            title_text = "Ensemble vs Data MMD^2"
        elif metric_name == 'kl':
            title_text = "KL(data || model) Divergence"
        elif metric_name == 'tvd':
            title_text = "Total Variation Distance"
        else:
            title_text = metric_name.replace('_', ' ').title()
        
        # Plot metric progression
        ax.plot(steps, metric_values, marker + '-', linewidth=2, markersize=8,
               label=f'Ensemble {metric_name.replace("_", " ").title()}', color=color)
        
        # Add baseline reference line
        baseline_label = f'Baseline ({baseline_value:.4f})' if scale_factor == 1 else f'Baseline ({baseline_value:.1f}%)'
        ax.axhline(y=baseline_value, color='red', linestyle='--',
                  label=baseline_label, alpha=0.6)
        
        ax.set_xlabel("Boosting step")
        ax.set_ylabel(ylabel)
        ax.set_title(title_text)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plot_path = output_manager.get_path('metrics_progression.pdf')
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Saved metrics progression plot to: {plot_path}")
    plt.close()


# =============================================================================
# Logging and Output Management
# =============================================================================

class Logger:
    """Logger that writes to both console and file."""
    
    def __init__(self, log_file: Path, mode: str = 'w'):
        self.log_file = log_file
        self.terminal = sys.stdout
        
        # Create log file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_file, mode, buffering=1)  # Line buffered
    
    def write(self, message):
        """Write to both terminal and file."""
        self.terminal.write(message)
        self.file.write(message)
    
    def flush(self):
        """Flush both streams."""
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        """Close file handle."""
        if hasattr(self, 'file') and self.file:
            self.file.close()


class OutputManager:
    """Manages output directory and logging for a single run."""
    
    def __init__(self, base_dir: str = 'out', run_name: str = None,
                 log_dir: str | Path | None = None, log_filename: str = 'log.txt',
                 append_log: bool = False):
        """
        Initialize output manager.
        
        Args:
            base_dir: Base output directory
            run_name: Optional custom run name (defaults to timestamp)
            log_dir: Optional directory for log file. If None, uses run directory.
            log_filename: Log filename.
            append_log: Whether to append instead of overwrite.
        """
        self.base_dir = Path(base_dir)
        
        # Create unique run directory with timestamp
        if run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"run_{timestamp}"
        
        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.log_file = (Path(log_dir) if log_dir is not None else self.run_dir) / log_filename
        log_mode = 'a' if append_log else 'w'
        self.logger = Logger(self.log_file, mode=log_mode)
        self.original_stdout = sys.stdout
        
        print(f"Output directory: {self.run_dir}")
        print(f"Log file: {self.log_file}")
        print("="*80)
    
    def __enter__(self):
        """Redirect stdout to logger."""
        sys.stdout = self.logger
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore stdout and close logger."""
        sys.stdout = self.original_stdout
        self.logger.close()
        
        if exc_type is None:
            print(f"\nRun completed. Results saved to: {self.run_dir}")
        else:
            print(f"\nRun failed with error. Log saved to: {self.log_file}")
    
    def get_path(self, filename: str) -> Path:
        """Get full path for a file in the run directory."""
        return self.run_dir / filename

    def save_config(self, config: dict):
        """Save configuration to JSON."""
        path = self.get_path('config.json')
        print(f"Saving config to {path}...")
        save_json(config, path)

    def save_metrics(self, metrics: dict, filename: str = 'metrics.json'):
        """Save metrics to JSON."""
        path = self.get_path(filename)
        print(f"Saving metrics to {path}...")
        save_json(metrics, path)
    
    def save_results_csv(self, metrics_history: dict, baseline_stats: dict = None, filename: str = 'results.csv'):
        """Save metrics history to CSV, including optional baseline and alpha."""
        path = self.get_path(filename)
        print(f"Saving results to {path}...")
        import csv
        
        # Determine all unique keys (excluding 'step' which we handle separately)
        all_keys = set(metrics_history.keys())
        if baseline_stats:
            all_keys.update(baseline_stats.keys())
        all_keys.discard('step')  # Remove 'step' since we prepend it separately
        
        # Sort keys alphabetically
        sorted_keys = sorted(list(all_keys))
        header = ['step'] + sorted_keys
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            # 1. Baseline row (step -1)
            if baseline_stats:
                row = [-1]
                for k in sorted_keys:
                    if k == 'alpha':
                        row.append(1.0)
                    elif k in baseline_stats:
                        row.append(baseline_stats[k])
                    else:
                        row.append('')
                writer.writerow(row)
            
            # 2. History rows
            n_steps = len(metrics_history[list(metrics_history.keys())[0]])
            for i in range(n_steps):
                row = [i]
                for k in sorted_keys:
                    if k in metrics_history:
                        val = metrics_history[k][i]
                        # Handle JAX/NumPy types
                        if hasattr(val, 'item'): val = val.item()
                        row.append(val)
                    else:
                        row.append('')
                writer.writerow(row)


def jax_json_serializer(obj):
    """JSON serializer for JAX/NumPy types."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(data: dict, path: Path):
    """Save dictionary to JSON with JAX support."""
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=jax_json_serializer)

def report_loss_components(data_mmd, ensemble_mmd, total_loss):
    """Print loss components for boosting step."""
    print(f"  Loss components: MMD_data={data_mmd:.6f}, MMD_ens={ensemble_mmd:.6f}, TOTAL={total_loss:.6f}")



