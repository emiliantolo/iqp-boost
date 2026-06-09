"""Plots for best-config retrain summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXCLUDED_METRICS = {"validity", "coverage", "precision", "recall", "support_match", "f_score"}
GROUPS = ["final_stats", "baseline_stats", "ensemble_fcfw_stats"]
GROUP_LABELS = ["Ensemble", "Baseline", "Ensemble (FCFW)"]
COLORS = {"final_stats": "#2ca02c", "baseline_stats": "#7f7f7f", "ensemble_fcfw_stats": "#1f77b4"}


def plot_best_retrain_summary(
    summary: dict,
    hpo_dir: Path,
    metric_filter: str | None = None,
    include_weight_distribution: bool = True,
) -> dict[str, str]:
    """Generate comparison plots from a best-retrains summary."""
    output_paths: dict[str, str] = {}
    output_paths.update(plot_metrics_comparison(summary, hpo_dir, metric_filter=metric_filter))
    output_paths.update(plot_metric_trends(summary, hpo_dir, metric_filter=metric_filter))
    if include_weight_distribution:
        output_paths.update(plot_weight_distribution(summary, hpo_dir))
    return output_paths


def _plot_metrics(summary: dict, metric_filter: str | None = None) -> list[str]:
    metrics = set()
    for group in GROUPS:
        metrics.update(summary.get("aggregates", {}).get(group, {}).keys())
    if metric_filter is not None:
        metric_filter = metric_filter.lower()
    return sorted(
        metric
        for metric in metrics
        if metric not in EXCLUDED_METRICS
        and (metric_filter is None or metric_filter in metric.lower())
    )


def _metric_value(seed: dict, group: str, metric: str) -> float:
    stats = seed.get(group) or {}
    value = stats.get(metric)
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return np.nan


def _save_figure(fig, path_base: Path) -> dict[str, str]:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path_base.with_suffix(".pdf")
    png_path = path_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return {pdf_path.stem + "_pdf": str(pdf_path), png_path.stem + "_png": str(png_path)}


def plot_metrics_comparison(
    summary: dict,
    hpo_dir: Path,
    metric_filter: str | None = None,
) -> dict[str, str]:
    """Plot aggregate Baseline vs Ensemble vs Ensemble(FCFW) metrics."""
    plot_metrics = _plot_metrics(summary, metric_filter=metric_filter)
    if not plot_metrics:
        return {}

    aggregates = summary.get("aggregates", {})
    n_cols = max(1, (len(plot_metrics) + 1) // 2)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    axes = np.asarray(axes).reshape(-1)

    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]
        positions = np.arange(len(GROUPS))
        means = []
        stds = []
        colors = []
        for group in GROUPS:
            metric_stats = aggregates.get(group, {}).get(metric)
            means.append(float(metric_stats["mean"]) if metric_stats else 0.0)
            stds.append(float(metric_stats["std"]) if metric_stats else 0.0)
            colors.append(COLORS[group])

        ax.bar(positions, means, 0.45, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.errorbar(positions, means, yerr=stds, fmt="none", color="black", capsize=4, linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(GROUP_LABELS, fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y")

    for idx in range(len(plot_metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("HPO Best-Retrain Metrics (mean +/- std across seeds)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, hpo_dir / "plots" / "metrics_comparison")


def plot_metric_trends(
    summary: dict,
    hpo_dir: Path,
    metric_filter: str | None = None,
) -> dict[str, str]:
    """Plot per-seed metric values for each available metric."""
    seeds = summary.get("seeds", [])
    plot_metrics = _plot_metrics(summary, metric_filter=metric_filter)
    if not seeds or not plot_metrics:
        return {}

    n_cols = 3
    n_rows = (len(plot_metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    seed_indices = list(range(len(seeds)))
    markers = {"final_stats": "o", "baseline_stats": "s", "ensemble_fcfw_stats": "^"}

    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]
        for group, label in zip(GROUPS, GROUP_LABELS, strict=True):
            values = [_metric_value(seed, group, metric) for seed in seeds]
            if any(np.isfinite(value) for value in values):
                ax.plot(
                    seed_indices,
                    values,
                    label=label,
                    color=COLORS[group],
                    marker=markers[group],
                    linewidth=1.5,
                    markersize=6,
                    alpha=0.85,
                )
        ax.set_xticks(seed_indices)
        ax.set_xticklabels([f"Seed {idx}" for idx in seed_indices])
        ax.set_title(metric, fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")
        ax.legend(fontsize=7, loc="best")

    for idx in range(len(plot_metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Per-Seed Metric Values", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, hpo_dir / "plots" / "metric_trends")


def plot_weight_distribution(summary: dict, hpo_dir: Path) -> dict[str, str]:
    """Plot ensemble and optional FCFW weights across retrain seeds."""
    seeds = summary.get("seeds", [])
    weight_data = [
        {
            "seed_index": seed.get("seed_index", idx),
            "weights": seed.get("weights"),
            "ensemble_fcfw_weights": seed.get("ensemble_fcfw_weights"),
        }
        for idx, seed in enumerate(seeds)
        if seed.get("weights") is not None
    ]
    if not weight_data:
        return {}

    n_models = len(weight_data[0]["weights"])
    x = np.arange(n_models)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for item in weight_data:
        axes[0].plot(x, item["weights"], marker="o", linewidth=1.2, label=f"Seed {item['seed_index']}")
    axes[0].axhline(y=1.0 / n_models, color="red", linestyle="--", alpha=0.5, label=f"Uniform (1/{n_models})")
    axes[0].set_xlabel("Model index")
    axes[0].set_ylabel("Weight")
    axes[0].set_title("Ensemble Weights")
    axes[0].grid(True, alpha=0.2, axis="y")
    axes[0].legend(fontsize=7, ncol=2)

    plotted_fcfw = False
    for item in weight_data:
        fcfw_weights = item.get("ensemble_fcfw_weights")
        if fcfw_weights is not None and len(fcfw_weights) == n_models:
            axes[1].plot(x, fcfw_weights, marker="o", linewidth=1.2, label=f"Seed {item['seed_index']}")
            plotted_fcfw = True
    if plotted_fcfw:
        axes[1].set_xlabel("Model index")
        axes[1].set_ylabel("Weight")
        axes[1].set_title("Ensemble Weights (FCFW)")
        axes[1].grid(True, alpha=0.2, axis="y")
        axes[1].legend(fontsize=7, ncol=2)
    else:
        axes[1].text(0.5, 0.5, "No FCFW weights available", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()

    fig.suptitle("Ensemble Weight Distribution Across Seeds", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, hpo_dir / "plots" / "weight_distribution")
