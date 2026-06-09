"""Generate bar charts with error bars from HPO best-retrains summary.

Reads best_retrains/summary.json and produces comparison plots
showing Baseline vs Ensemble vs Ensemble(FCFW) with mean +/- std
across seeds for all available metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary(summary_path: str) -> dict:
    return json.loads(Path(summary_path).read_text())


def _metric_values(seeds: list[dict], group: str, metric: str) -> float:
    for seed in seeds:
        stats = seed.get(group, {})
        if stats is None:
            continue
        val = stats.get(metric)
        if val is not None and isinstance(val, (int, float)) and np.isfinite(val):
            return float(val)
    return np.nan


def plot_metrics_comparison(summary: dict, output_dir: Path) -> None:
    seeds = summary["seeds"]
    aggregates = summary["aggregates"]

    groups = ["final_stats", "baseline_stats", "ensemble_fcfw_stats"]
    group_labels = ["Ensemble", "Baseline", "Ensemble (FCFW)"]

    # Collect all metric keys from all groups
    all_metrics = set()
    for group in groups:
        if group in aggregates:
            all_metrics.update(aggregates[group].keys())
    all_metrics = sorted(all_metrics)

    # Filter to numeric metrics (exclude non-quantitative ones)
    plot_metrics = [m for m in all_metrics if m not in ("validity", "coverage", "precision", "recall", "support_match", "f_score")]

    n_metrics = len(plot_metrics)
    if n_metrics == 0:
        print("No numeric metrics to plot.")
        return

    n_cols = (n_metrics + 1) // 2
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    group_means = {}
    group_stds = {}
    for group in groups:
        if group in aggregates:
            group_means[group] = {m: aggregates[group][m]["mean"] for m in plot_metrics if m in aggregates[group]}
            group_stds[group] = {m: aggregates[group][m]["std"] for m in plot_metrics if m in aggregates[group]}

    colors = {"final_stats": "#2ca02c", "baseline_stats": "#7f7f7f", "ensemble_fcfw_stats": "#1f77b4"}

    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]
        x = np.arange(len(groups))
        width = 0.22

        bar_means = []
        bar_stds = []
        bar_colors = []
        bar_positions = []

        for gi, group in enumerate(groups):
            if group in group_means and metric in group_means[group]:
                mean = group_means[group][metric]
                std = group_stds[group][metric]
                bar_means.append(mean)
                bar_stds.append(std)
                bar_colors.append(colors[group])
                bar_positions.append(x[gi])
            else:
                bar_means.append(0)
                bar_stds.append(0)
                bar_colors.append(colors[group])
                bar_positions.append(x[gi])

        ax.bar(bar_positions, bar_means, width, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.errorbar(bar_positions, bar_means, yerr=bar_stds, fmt="none", color="black", capsize=4, capthick=0.8, linewidth=0.8)

        ax.set_xticks(list(x))
        ax.set_xticklabels(group_labels, fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y")

        # Add value labels on bars
        max_std = max(bar_stds) if bar_stds else 0.001
        for i, (pos, mean) in enumerate(zip(bar_positions, bar_means)):
            if mean != 0:
                ax.text(pos, mean + bar_stds[i] + max_std * 0.05, f"{mean:.3f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Hide unused subplots
    for idx in range(len(plot_metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("HPO Best-Retrain Metrics (mean +/- std across seeds)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = output_dir / "plots" / "metrics_comparison.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics comparison plot to: {out_path}")

    # Also save as PNG for quick viewing
    out_png = output_dir / "plots" / "metrics_comparison.png"
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]
        x = np.arange(len(groups))
        width = 0.22

        bar_means = []
        bar_stds = []
        bar_colors = []
        bar_positions = []

        for gi, group in enumerate(groups):
            if group in group_means and metric in group_means[group]:
                mean = group_means[group][metric]
                std = group_stds[group][metric]
                bar_means.append(mean)
                bar_stds.append(std)
                bar_colors.append(colors[group])
                bar_positions.append(x[gi])
            else:
                bar_means.append(0)
                bar_stds.append(0)
                bar_colors.append(colors[group])
                bar_positions.append(x[gi])

        ax.bar(bar_positions, bar_means, width, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.errorbar(bar_positions, bar_means, yerr=bar_stds, fmt="none", color="black", capsize=4, capthick=0.8, linewidth=0.8)

        ax.set_xticks(list(x))
        ax.set_xticklabels(group_labels, fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y")

    for idx in range(len(plot_metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("HPO Best-Retrain Metrics (mean +/- std across seeds)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved metrics comparison plot to: {out_png}")


def plot_metric_trends(summary: dict, output_dir: Path) -> None:
    """Plot per-seed metric values as connected points to show variation."""
    seeds = summary["seeds"]
    aggregates = summary["aggregates"]

    groups = ["final_stats", "baseline_stats", "ensemble_fcfw_stats"]
    group_labels = ["Ensemble", "Baseline", "Ensemble (FCFW)"]

    all_metrics = set()
    for group in groups:
        if group in aggregates:
            all_metrics.update(aggregates[group].keys())
    all_metrics = sorted(all_metrics)
    plot_metrics = [m for m in all_metrics if m not in ("validity", "coverage", "precision", "recall", "support_match", "f_score")]

    n_metrics = len(plot_metrics)
    if n_metrics == 0:
        return

    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    colors = {"final_stats": "#2ca02c", "baseline_stats": "#7f7f7f", "ensemble_fcfw_stats": "#1f77b4"}
    markers = {"final_stats": "o", "baseline_stats": "s", "ensemble_fcfw_stats": "^"}

    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]
        seed_indices = list(range(len(seeds)))

        for gi, group in enumerate(groups):
            values = [_metric_values([seeds[s]], group, metric) for s in seed_indices]
            if any(np.isfinite(v) for v in values):
                finite_vals = [v if np.isfinite(v) else np.nan for v in values]
                ax.plot(seed_indices, finite_vals, label=group_labels[gi],
                        color=colors[group], marker=markers[group], linewidth=1.5, markersize=6, alpha=0.85)

        ax.set_xticks(seed_indices)
        ax.set_xticklabels([f"Seed {i}" for i in seed_indices])
        ax.set_title(metric, fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")
        if len(groups) <= 3:
            ax.legend(fontsize=7, loc="best")

    for idx in range(len(plot_metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Per-Seed Metric Values", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = output_dir / "plots" / "metric_trends.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric trends plot to: {out_path}")

    out_png = output_dir / "plots" / "metric_trends.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved metric trends plot to: {out_png}")


def plot_weight_distribution(summary: dict, output_dir: Path) -> None:
    """Plot Frank-Wolfe ensemble weights across seeds."""
    seeds = summary["seeds"]
    n_models = None
    weight_data = []

    for seed in seeds:
        weights = seed.get("weights")
        fcfw_weights = seed.get("ensemble_fcfw_weights")
        if weights is not None:
            if n_models is None:
                n_models = len(weights)
            weight_data.append({"seed": seed["seed_index"], "weights": weights, "fcfw": fcfw_weights})

    if not weight_data or n_models is None:
        print("No weight data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Uniform weights (initial FW)
    ax1 = axes[0]
    x = np.arange(n_models)
    width = 0.35
    for wd in weight_data:
        ax1.bar(x - width/2, [wd["weights"][i] for i in range(n_models)],
                width, label=f"Seed {wd['seed']}", alpha=0.7, color=f"C{wd['seed']}")
    ax1.axhline(y=1.0/n_models, color="red", linestyle="--", alpha=0.5, label=f"Uniform (1/{n_models})")
    ax1.set_xlabel("Model index")
    ax1.set_ylabel("Weight")
    ax1.set_title("Ensemble Weights (Frank-Wolfe)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"M{i}" for i in range(n_models)])
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.2, axis="y")

    # FCFW weights
    ax2 = axes[1]
    has_fcfw = any(wd.get("fcfw") is not None for wd in weight_data)
    if has_fcfw:
        for wd in weight_data:
            fcfw = wd.get("fcfw", [])
            if fcfw and len(fcfw) == n_models:
                ax2.bar(x + width/2, fcfw, width, label=f"Seed {wd['seed']}", alpha=0.7, color=f"C{wd['seed']}")
        ax2.set_xlabel("Model index")
        ax2.set_ylabel("Weight")
        ax2.set_title("Ensemble Weights (FCFW)")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"M{i}" for i in range(n_models)])
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.2, axis="y")
    else:
        ax2.text(0.5, 0.5, "No FCFW weights available", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()

    fig.suptitle("Ensemble Weight Distribution Across Seeds", fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()

    out_path = output_dir / "plots" / "weight_distribution.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved weight distribution plot to: {out_path}")

    out_png = output_dir / "plots" / "weight_distribution.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved weight distribution plot to: {out_png}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/hpo_retrains_plots.py <hpo_output_dir>")
        print("  e.g.: python scripts/hpo_retrains_plots.py out/hpo/hamming_balls_20q_k4_p008_heavy_hex_20260605_100632")
        sys.exit(1)

    hpo_dir = Path(sys.argv[1])
    summary_path = hpo_dir / "best_retrains" / "summary.json"

    if not summary_path.exists():
        print(f"Summary not found: {summary_path}")
        sys.exit(1)

    summary = load_summary(str(summary_path))
    n_seeds = summary["n_seeds"]
    print(f"Generating plots from {n_seeds} seeds: {summary_path}")

    plot_metrics_comparison(summary, hpo_dir)
    plot_metric_trends(summary, hpo_dir)
    plot_weight_distribution(summary, hpo_dir)

    print("Done.")


if __name__ == "__main__":
    main()
