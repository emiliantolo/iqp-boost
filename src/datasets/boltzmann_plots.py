"""Plot builders for Boltzmann-style dataset integrations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .boltzmann_metrics import (
    samples_to_hamming_weights,
    spin_covariance_from_probs,
    spin_covariance_from_samples,
    covariance_matrices,
)


def _finalize(fig, title: str | None = None):
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_hamming_weight_histogram(reference_samples, baseline_samples, ensemble_samples):
    """Side-by-side histogram comparing ground truth, baseline, and ensemble Hamming weights."""
    reference_weights = samples_to_hamming_weights(reference_samples)
    baseline_weights = samples_to_hamming_weights(baseline_samples)
    ensemble_weights = samples_to_hamming_weights(ensemble_samples)

    n_qubits = int(max(
        reference_weights.max(initial=0),
        baseline_weights.max(initial=0),
        ensemble_weights.max(initial=0),
    ))
    bins = np.arange(0, n_qubits + 2) - 0.5

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    sns.histplot(reference_weights, bins=bins, discrete=True, stat='probability', ax=axes[0], color='black')
    sns.histplot(baseline_weights, bins=bins, discrete=True, stat='probability', ax=axes[1], color='#7f7f7f')
    sns.histplot(ensemble_weights, bins=bins, discrete=True, stat='probability', ax=axes[2], color='#2ca02c')
    axes[0].set_title('Ground Truth Hamming Weights')
    axes[1].set_title('Baseline Hamming Weights')
    axes[2].set_title('Ensemble Hamming Weights')
    for ax in axes:
        ax.set_xlabel('Hamming weight')
        ax.set_ylabel('Probability')
        ax.grid(True, alpha=0.2)

    return _finalize(fig, 'Hamming Weight Comparison')


def plot_covariance_heatmaps(sigma_reference, sigma_baseline, sigma_ensemble):
    """Heatmaps of the ground truth, baseline, and ensemble pairwise spin covariance matrices."""
    sigma_reference = np.asarray(sigma_reference, dtype=np.float64)
    sigma_baseline = np.asarray(sigma_baseline, dtype=np.float64)
    sigma_ensemble = np.asarray(sigma_ensemble, dtype=np.float64)
    vmin = float(min(np.min(sigma_reference), np.min(sigma_baseline), np.min(sigma_ensemble)))
    vmax = float(max(np.max(sigma_reference), np.max(sigma_baseline), np.max(sigma_ensemble)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    sns.heatmap(sigma_reference, ax=axes[0], cmap='vlag', vmin=vmin, vmax=vmax, square=True, cbar=True)
    sns.heatmap(sigma_baseline, ax=axes[1], cmap='vlag', vmin=vmin, vmax=vmax, square=True, cbar=True)
    sns.heatmap(sigma_ensemble, ax=axes[2], cmap='vlag', vmin=vmin, vmax=vmax, square=True, cbar=True)
    axes[0].set_title('Ground Truth Covariance')
    axes[1].set_title('Baseline Covariance')
    axes[2].set_title('Ensemble Covariance')
    return fig


def _sorted_cdf_from_samples(samples: np.ndarray, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    if samples is None or len(samples) == 0:
        probs = np.zeros(2 ** n_qubits, dtype=np.float64)
        return np.sort(probs)[::-1], np.cumsum(np.sort(probs)[::-1])
    indices = np.sum(samples.astype(int) * (2 ** np.arange(n_qubits)), axis=1)
    counts = np.bincount(indices, minlength=2 ** n_qubits)
    probs = counts.astype(np.float64)
    total = float(probs.sum())
    if total > 0:
        probs /= total
    sorted_p = np.sort(probs)[::-1]
    return sorted_p, np.cumsum(sorted_p)


def _sorted_cdf_from_probs(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return p, p
    sorted_p = np.sort(p)[::-1]
    return sorted_p, np.cumsum(sorted_p)


def plot_lorenz_curve(
    exact_probs: np.ndarray | None,
    baseline_samples,
    ensemble_samples,
    reference_samples=None,
):
    """Lorenz / CDF visualization for exact or empirical distributions.

    Left panel: cumulative probability mass vs number of modes covered.
    Right panel: sorted probability spectrum (exact vs empirical).
    """

    def _samples_to_empirical(samples, n_states, n_qubits):
        if samples is None or len(samples) == 0:
            probs = np.zeros(n_states, dtype=np.float64)
            return probs, np.cumsum(np.sort(probs)[::-1])
        indices = np.sum(samples.astype(int) * (2 ** np.arange(n_qubits)), axis=1)
        counts = np.bincount(indices, minlength=n_states)
        probs = counts / counts.sum()
        sorted_p = np.sort(probs)[::-1]
        return sorted_p, np.cumsum(sorted_p)

    def _participation_ratio(probs: np.ndarray) -> float:
        probs = np.asarray(probs, dtype=np.float64)
        denom = float(np.sum(probs ** 2))
        if denom <= 0.0:
            return float('nan')
        return float(1.0 / denom)

    if exact_probs is not None:
        exact_probs = np.asarray(exact_probs, dtype=np.float64)
        p_reference = exact_probs / float(max(exact_probs.sum(), 1e-300))
        n_qubits = int(round(np.log2(len(p_reference))))
        reference_label = 'Exact Boltzmann'
    elif reference_samples is not None and len(reference_samples) > 0:
        n_qubits = reference_samples.shape[1]
        n_states = 2 ** n_qubits
        p_reference, _ = _samples_to_empirical(reference_samples, n_states, n_qubits)
        reference_label = 'Reference training-empirical'
    elif baseline_samples is not None and len(baseline_samples) > 0:
        n_qubits = baseline_samples.shape[1]
        n_states = 2 ** n_qubits
        p_reference, _ = _samples_to_empirical(baseline_samples, n_states, n_qubits)
        reference_label = 'Baseline reference'
    elif ensemble_samples is not None and len(ensemble_samples) > 0:
        n_qubits = ensemble_samples.shape[1]
        n_states = 2 ** n_qubits
        p_reference, _ = _samples_to_empirical(ensemble_samples, n_states, n_qubits)
        reference_label = 'Ensemble reference'
    else:
        raise ValueError('At least one of exact_probs, reference_samples, baseline_samples, or ensemble_samples must be provided')

    p_sorted = np.sort(p_reference)[::-1]
    cdf_reference = np.cumsum(p_sorted)
    k = np.arange(1, len(p_sorted) + 1)
    pr_reference = _participation_ratio(p_reference)
    pr_uniform = float(len(p_reference))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(k, cdf_reference, label=reference_label, color='black', linewidth=2)

    if baseline_samples is not None and len(baseline_samples) > 0:
        baseline_sorted, baseline_cdf = _sorted_cdf_from_samples(baseline_samples, n_qubits)
        ax.plot(k, baseline_cdf, label='Baseline', color='#7f7f7f', linewidth=1.6, linestyle='--')
    if ensemble_samples is not None and len(ensemble_samples) > 0:
        ensemble_sorted, ensemble_cdf = _sorted_cdf_from_samples(ensemble_samples, n_qubits)
        ax.plot(k, ensemble_cdf, label='Ensemble', color='#2ca02c', linewidth=1.8, alpha=0.9)
    else:
        ensemble_sorted = None

    for threshold in [0.90, 0.99]:
        idx = int(np.searchsorted(cdf_reference, threshold))
        if idx < len(k):
            ax.plot(idx + 1, cdf_reference[idx], 'o', markersize=5, color='red')
            ax.annotate(
                f'k={idx + 1}',
                (idx + 1, cdf_reference[idx]),
                textcoords='offset points',
                xytext=(5, -12),
                fontsize=8,
                color='red',
            )

    ax.set_xlabel('Number of modes covered (k)')
    ax.set_ylabel('Cumulative probability mass')
    ax.set_title('Lorenz curve')
    ax.set_xscale('log')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(k))
    ax.set_ylim(0, 1.05)

    ax2 = axes[1]
    ax2.plot(k, p_sorted, label=reference_label, color='black', linewidth=1.5)
    ax2.axhline(1.0 / len(p_sorted), color='gray', linestyle='--', alpha=0.5, label='Uniform')
    if baseline_samples is not None and len(baseline_samples) > 0:
        ax2.plot(k, baseline_sorted, label='Baseline', color='#7f7f7f', linewidth=1.5, linestyle='--')
    if ensemble_sorted is not None:
        ax2.plot(k, ensemble_sorted, label='Ensemble', color='#2ca02c', linewidth=1.5, alpha=0.8)

    ax2.set_xlabel('State rank')
    ax2.set_ylabel('Probability')
    ax2.set_title('Probability spectrum')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(k))
    ax2.text(
        0.03,
        0.03,
        "\n".join([
            f"PR exact: {pr_reference:.1f}",
            f"PR uniform: {pr_uniform:.0f}",
        ]),
        transform=ax2.transAxes,
        fontsize=8,
        va='bottom',
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.65, 'edgecolor': 'gray'},
    )

    return _finalize(fig)


def plot_boosting_convergence(loss_history, metric_history):
    """Plot MMD loss and correlation Frobenius error over boosting iterations."""

    def _extract(series, key=None):
        if isinstance(series, dict):
            if key is None:
                return np.asarray(series.get('loss', series.get('mmd', [])), dtype=np.float64)
            return np.asarray(series.get(key, []), dtype=np.float64)
        return np.asarray(series, dtype=np.float64)

    losses = _extract(loss_history)
    if isinstance(metric_history, dict):
        corr = _extract(metric_history, 'corr_fro')
        steps = np.asarray(metric_history.get('step', np.arange(max(len(losses), len(corr)))) , dtype=np.float64)
    else:
        corr = _extract(metric_history)
        steps = np.arange(max(len(losses), len(corr)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(np.arange(len(losses)), losses, color='#1f77b4', marker='o')
    axes[0].set_ylabel('MMD loss')
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(steps[:len(corr)], corr, color='#d62728', marker='s')
    axes[1].set_xlabel('Boosting iteration')
    axes[1].set_ylabel('Correlation Frobenius error')
    axes[1].grid(True, alpha=0.2)

    return _finalize(fig, 'Boosting Convergence')


def boltzmann_summary_figures(
    reference_samples,
    baseline_samples,
    ensemble_samples,
    exact_probs: np.ndarray | None = None,
):
    """Create a compact set of Boltzmann evaluation figures.

    Returns a dict with optional figures keyed by the plot type.
    """
    if exact_probs is not None:
        sigma_reference = spin_covariance_from_probs(exact_probs)
    else:
        sigma_reference = spin_covariance_from_samples(reference_samples)

    figures = {
        'hamming': plot_hamming_weight_histogram(reference_samples, baseline_samples, ensemble_samples),
    }

    sigma_baseline, sigma_ensemble = covariance_matrices(baseline_samples, ensemble_samples, exact_probs=exact_probs)
    figures['covariance'] = plot_covariance_heatmaps(sigma_reference, sigma_baseline, sigma_ensemble)

    figures['lorenz'] = plot_lorenz_curve(exact_probs, baseline_samples, ensemble_samples)

    return figures
