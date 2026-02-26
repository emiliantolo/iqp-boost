"""Unified reporting utilities for ensemble boosting."""

import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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


def report_step(step: int, n_models: int, ensemble_mmd: float, data_loss: float = None, ens_loss: float = None):
    """Report single boosting step result."""
    parts = [f"[Step {step}/{n_models-1}]"]
    if data_loss is not None:
        parts.append(f"data={data_loss:.4f}")
    if ens_loss is not None:
        parts.append(f"ens={ens_loss:.4f}")
    parts.append(f"MMD={ensemble_mmd:.4f}")
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


def report_metrics_table(baseline_stats: dict, ensemble_stats: dict, model_list: list = None, title: str = "FINAL METRICS"):
    """
    Report unified metrics table with all metrics (MMD, KL, validity, coverage, precision, recall, F1).
    
    Args:
        baseline_stats: Dict with baseline metrics
        ensemble_stats: Dict with ensemble metrics
        model_list: List of (name, stats_dict) tuples for individual models (optional)
        title: Table title
    """
    # Show all unified metrics
    headers = ["Model", "MMD", "KL", "Valid", "Cover", "Prec", "Recall", "F1"]
    rows = [
        ["Baseline", 
         f"{baseline_stats['mmd']:.4f}",
         f"{baseline_stats.get('kl', 0):.3f}",
         f"{100*baseline_stats.get('validity', 0):.1f}%",
         f"{100*baseline_stats.get('coverage', 0):.1f}%",
         f"{100*baseline_stats.get('precision', 0):.1f}%",
         f"{100*baseline_stats.get('recall', 0):.1f}%",
         f"{100*baseline_stats.get('f_score', 0):.1f}%"],
        ["Ensemble", 
         f"{ensemble_stats['mmd']:.4f}",
         f"{ensemble_stats.get('kl', 0):.3f}",
         f"{100*ensemble_stats.get('validity', 0):.1f}%",
         f"{100*ensemble_stats.get('coverage', 0):.1f}%",
         f"{100*ensemble_stats.get('precision', 0):.1f}%",
         f"{100*ensemble_stats.get('recall', 0):.1f}%",
         f"{100*ensemble_stats.get('f_score', 0):.1f}%"],
    ]
    if model_list:
        for name, stats in model_list:
            rows.append([name,
                        f"{stats['mmd']:.4f}",
                        f"{stats.get('kl', 0):.3f}",
                        f"{100*stats.get('validity', 0):.1f}%",
                        f"{100*stats.get('coverage', 0):.1f}%",
                        f"{100*stats.get('precision', 0):.1f}%",
                        f"{100*stats.get('recall', 0):.1f}%",
                        f"{100*stats.get('f_score', 0):.1f}%"])
    
    print(f"\n{title}")
    header_line = " | ".join(f"{h:<10}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        row_line = " | ".join(f"{str(v):<10}" for v in row)
        print(row_line)


def plot_data_ensemble_loss(ensemble, ensemble_metrics_history, baseline_stats, output_manager,
                           baseline_train_losses=None, accepted_steps=None, 
                           all_mmd_values=None, all_step_positions=None):
    """
    Plot data and ensemble loss history (epoch-level).
    
    This function creates a 2-panel plot:
    - Left: Baseline training curve (separate) + model_0 point + ensemble MMD progression (show rejected as X)
    - Right: Data, ensemble, and dual loss components across training with step separators
    
    Args:
        ensemble: BoostedEnsemble object with training_losses attribute
        ensemble_metrics_history: Dict with 'mmd' key containing step-level MMD values
        baseline_stats: Dict with baseline metrics including 'mmd'
        output_manager: OutputManager instance for saving plots
        baseline_train_losses: Optional array of baseline training losses (if trained separately)
        accepted_steps: List of step indices that were accepted (skip others in ensemble MMD plot)
        all_mmd_values: List of all MMD values (both accepted and rejected) for each trained model
        all_step_positions: List of step numbers for each trained model (to identify rejected ones)
    """
    if not hasattr(ensemble, 'training_losses') or len(ensemble.training_losses) == 0:
        print("No training losses available for plotting")
        return
    
    if accepted_steps is None:
        accepted_steps = list(range(len(ensemble_metrics_history['mmd'])))
    
    if all_mmd_values is None:
        all_mmd_values = []
    
    if all_step_positions is None:
        all_step_positions = []
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Left panel: Baseline (separate) + Model_0 + ensemble progression
    ax = axes[0]
    cumulative_epochs = 0
    
    # Plot baseline training curve if provided (separate training)
    if baseline_train_losses is not None and len(baseline_train_losses) > 0:
        baseline_epochs = np.arange(len(baseline_train_losses))
        ax.plot(baseline_epochs, baseline_train_losses, '-', linewidth=1.5, alpha=0.7,
                color='gray', label='Baseline training (separate)', zorder=1)
        
        # Mark baseline final evaluation
        ax.plot(len(baseline_train_losses) - 1, baseline_stats['mmd'], 'o', markersize=10,
                color='gray', markeredgecolor='black', markeredgewidth=1.5,
                label='Baseline final eval', zorder=8)
    
    # Model_0 is ensemble.training_losses[0] - plot with same color as ensemble steps
    model_0_loss = ensemble.training_losses[0]['total']
    model_0_epochs = np.arange(len(model_0_loss))
    ax.plot(model_0_epochs, model_0_loss, '-', linewidth=1.5, alpha=0.5,
            color='blue', label='Model 0 training', zorder=2)
    model_0_final_mmd = all_mmd_values[0]
    model_0_final_epoch = len(model_0_loss) - 1
    ax.plot(model_0_final_epoch, model_0_final_mmd, 'o', markersize=10,
            color='blue', markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    
    cumulative_epochs += len(model_0_loss)
    
    # Mark ensemble MMD at end of each step
    # Plot both accepted (circles) and rejected (X) points using all_mmd_values
    step_positions_accepted = []
    step_mmds_accepted = []
    step_positions_rejected = []
    step_mmds_rejected = []
    
    # Iterate through all trained steps
    current_cumulative_epochs = cumulative_epochs
    for step_idx, step_num in enumerate(all_step_positions):
        # Advance cumulative epochs to end of this model's training
        current_cumulative_epochs += len(ensemble.training_losses[step_num]['total'])
        
        if step_idx + 1 < len(all_mmd_values):  # +1 because all_mmd_values[0] is model_0
            mmd_val = all_mmd_values[step_idx + 1]
            is_accepted = step_num in accepted_steps
            
            if is_accepted:
                step_positions_accepted.append(current_cumulative_epochs)
                step_mmds_accepted.append(mmd_val)
            else:
                step_positions_rejected.append(current_cumulative_epochs)
                step_mmds_rejected.append(mmd_val)
    
    # Plot accepted steps as blue circles with connecting line from model_0
    if len(step_positions_accepted) > 0:
        # Connect model_0 to all accepted steps as continuous line
        all_accepted_epochs = [model_0_final_epoch] + step_positions_accepted
        all_accepted_mmds = [model_0_final_mmd] + step_mmds_accepted
        ax.plot(all_accepted_epochs, all_accepted_mmds, 'o-', linewidth=2.5, markersize=10,
                color='blue', markeredgecolor='black', markeredgewidth=1.5,
                label='Ensemble MMD (accepted)', zorder=10)
    
    # Plot rejected steps as red X marks
    if len(step_positions_rejected) > 0:
        ax.plot(step_positions_rejected, step_mmds_rejected, 'x', markersize=12, markeredgewidth=2,
                color='red', label='Rejected (no improve)', zorder=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMD²")
    ax.set_title("Data MMD² Loss")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc='best')
    
    # Right panel: Data, ensemble, and dual loss components with separators
    ax = axes[1]
    if len(ensemble.training_losses) > 1:
        # Start from end of model_0 training (cumulative epoch position)
        model_0_epochs = len(ensemble.training_losses[0]['total'])
        cumulative_epochs = model_0_epochs  # Start from where model_1 begins
        step_boundaries = [model_0_epochs]  # First boundary at model_0 end
        
        # Color palette for models
        colors_data = plt.cm.Blues(np.linspace(0.5, 0.9, len(ensemble.training_losses) - 1))
        colors_ens = plt.cm.Oranges(np.linspace(0.5, 0.9, len(ensemble.training_losses) - 1))
        
        # Skip baseline (model 0), plot boosted models starting from model_1
        for i in range(1, len(ensemble.training_losses)):
            loss_dict = ensemble.training_losses[i]
            total_loss = loss_dict['total']  # Full resolution dual loss (logged at every epoch)
            epochs_full = np.arange(len(total_loss)) + cumulative_epochs
            
            # Plot EXACT dual loss (logged at every training step) as main curve
            ax.plot(epochs_full, total_loss, '-',
                    linewidth=2.5, markersize=0, alpha=0.8, color='black', zorder=5,
                    label=f'M{i+1} dual' if i < 4 or i == len(ensemble.training_losses)-1 else None)
            
            # Plot component estimates (recomputed at monitor_interval checkpoints)
            if 'data_history' in loss_dict and 'ensemble_history' in loss_dict:
                sampled_epochs = loss_dict['sampled_epochs'] + cumulative_epochs
                # Use absolute values to ensure data/ensemble losses are always positive
                data_history = np.abs(loss_dict['data_history'])
                ens_history = np.abs(loss_dict['ensemble_history'])
                
                # Plot data loss (attraction) - positive
                ax.plot(sampled_epochs, data_history, 'o-',
                        linewidth=1.5, markersize=3, alpha=0.6, color=colors_data[i-1],
                        label=f'M{i+1} data' if i < 4 or i == len(ensemble.training_losses)-1 else None)
                
                # Plot ensemble loss (repulsion) - shown as negative to oppose data loss
                ax.plot(sampled_epochs, -ens_history, 's-',
                        linewidth=1.5, markersize=3, alpha=0.6, color=colors_ens[i-1],
                        label=f'M{i+1} ens' if i < 4 or i == len(ensemble.training_losses)-1 else None)
                
            else:
                # Fallback: show constant component estimates from final values
                # (still plot exact dual loss at full resolution)
                if loss_dict['data_final'] is not None and loss_dict['ensemble_final'] is not None:
                    epochs = np.arange(len(total_loss)) + cumulative_epochs
                    data_final = abs(loss_dict['data_final'])
                    ens_final = abs(loss_dict['ensemble_final'])
                    
                    ax.plot(epochs, np.full_like(epochs, data_final, dtype=float), '--',
                            linewidth=1.5, alpha=0.6, color=colors_data[i-1],
                            label=f'M{i+1} data (est)' if i < 4 or i == len(ensemble.training_losses)-1 else None)
                    
                    ax.plot(epochs, np.full_like(epochs, -ens_final, dtype=float), '--',
                            linewidth=1.5, alpha=0.6, color=colors_ens[i-1],
                            label=f'M{i+1} ens (est)' if i < 4 or i == len(ensemble.training_losses)-1 else None)
            
            cumulative_epochs += len(total_loss)
            step_boundaries.append(cumulative_epochs)
        
        # Add subtle vertical separator lines for model boundaries
        for boundary in step_boundaries[1:-1]:  # Exclude first and last
            ax.axvline(x=boundary, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        
        ax.set_xlabel("Epoch (cumulative)")
        ax.set_ylabel("Loss")
        ax.set_title("Dual MMD Losses")
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
            ('mmd', 'MMD²', 1, 'blue', 'o'),
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
            title_text = "F1 Score (Gaussian Classification)"
        elif metric_name == 'coverage':
            title_text = "Ground Truth Coverage" if scale_factor > 1 else "Ground Truth Coverage"
        elif metric_name == 'mmd':
            title_text = "Ensemble MMD vs Ground Truth"
        elif metric_name == 'kl':
            title_text = "KL(data || model) Divergence"
        else:
            title_text = metric_name.replace('_', ' ').title()
        
        # Plot metric progression
        ax.plot(steps, metric_values, marker + '-', linewidth=2, markersize=8,
               label=f'Ensemble {metric_name.replace("_", " ").title()}', color=color)
        
        # Add baseline reference line
        baseline_label = f'Baseline ({baseline_value:.4f})' if scale_factor == 1 else f'Baseline ({baseline_value:.1f}%)'
        ax.axhline(y=baseline_value, color='red', linestyle='--',
                  label=baseline_label, alpha=0.6)
        
        ax.set_xlabel("Step")
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
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.terminal = sys.stdout
        
        # Create log file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_file, 'w', buffering=1)  # Line buffered
    
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
    
    def __init__(self, base_dir: str = 'out', run_name: str = None):
        """
        Initialize output manager.
        
        Args:
            base_dir: Base output directory
            run_name: Optional custom run name (defaults to timestamp)
        """
        self.base_dir = Path(base_dir)
        
        # Create unique run directory with timestamp
        if run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"run_{timestamp}"
        
        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        log_file = self.run_dir / 'log.txt'
        self.logger = Logger(log_file)
        self.original_stdout = sys.stdout
        
        print(f"Output directory: {self.run_dir}")
        print(f"Log file: {log_file}")
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
            print(f"\nRun failed with error. Logs saved to: {self.run_dir}")
    
    def get_path(self, filename: str) -> Path:
        """Get full path for a file in the run directory."""
        return self.run_dir / filename



