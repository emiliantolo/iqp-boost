"""Dataset and plotting factories for config-driven experiments."""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from src.datasets.bas import BarsAndStripesDataset
from src.datasets.blobs import BlobsDataset
from src.datasets.dwave import DWaveDataset
from src.datasets.gaussian import GaussianMixtureDataset
from src.datasets.genomic import GenomicDataset
from src.datasets.ising import FrustratedIsingDataset
from src.datasets.pennylane_ising import PennylaneIsingDataset
from src.datasets.pennylane_bas import PennylaneBASDataset
from src.datasets.pennylane_hm import PennylaneHMDataset
from src.datasets.mnist import BinarizedMNISTDataset
from src.datasets.noisy_bas import NoisyBASDataset
from src.datasets.parity import ParityDataset
from src.datasets.scale_free import ScaleFreeDataset
from src.datasets.shapes import BinaryShapesDataset
from src.datasets.random_circuit import RandomCircuitDataset
from src.datasets.qaoa_maxcut import QAOAMaxCutDataset
from src.datasets.tfim_thermal import TFIMThermalDataset
from src.datasets.rydberg import RydbergDataset


def _resolve_rows_cols(params: dict, config: dict, default: tuple[int, int] = (4, 4)) -> tuple[int, int]:
    """Resolve dataset dimensions from params/config.

    Accepts either:
    - params['rows'] / params['cols']
    - params['dims'] = [rows, cols]
    with fallback to config equivalents and then default.
    """
    if 'rows' in params and 'cols' in params:
        return int(params['rows']), int(params['cols'])

    dims = params.get('dims', None)
    if dims is None and 'rows' in config and 'cols' in config:
        return int(config['rows']), int(config['cols'])
    if dims is None:
        dims = config.get('dims', default)

    return int(dims[0]), int(dims[1])


def _binary_rows_to_ints(samples: np.ndarray) -> np.ndarray:
    if samples is None or len(samples) == 0:
        return np.array([], dtype=int)
    return np.array([int("".join(map(str, row.astype(int))), 2) for row in samples], dtype=object)


def _histogram_viz(title: str, out_filename: str, n_qubits: int) -> Callable:
    def _viz(output, x_train, baseline_samples, final_samples, per_model_samples, weights):
        if baseline_samples is None or final_samples is None:
            print("Skipping histogram visualization because sampling output is unavailable.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        gt_indices = _binary_rows_to_ints(x_train)
        baseline_indices = _binary_rows_to_ints(baseline_samples)
        ensemble_indices = _binary_rows_to_ints(final_samples)
        n_bins = min(128, 2 ** min(n_qubits, 10))

        axes[0].hist(gt_indices, bins=n_bins, color='blue', alpha=0.7, edgecolor='black')
        axes[0].set_title('Ground Truth')
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(baseline_indices, bins=n_bins, color='gray', alpha=0.7, edgecolor='black')
        axes[1].set_title('Baseline')
        axes[1].grid(True, alpha=0.3)

        axes[2].hist(ensemble_indices, bins=n_bins, color='green', alpha=0.7, edgecolor='black')
        axes[2].set_title('Final Ensemble')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()
        path = output.get_path(out_filename)
        plt.savefig(path, format='pdf', bbox_inches='tight')
        print(f"Saved histogram visualization to: {path}")
        plt.close(fig)

    return _viz


def _sample_grid_viz(dataset, title: str, out_filename: str,
                     n_models_to_show: int = 3, samples_per_model: int = 4) -> Callable:
    def _viz(output, x_train, baseline_samples, final_samples, per_model_samples, weights):
        if not per_model_samples:
            print("Skipping sample-grid visualization because per-model samples are unavailable.")
            return

        rows = min(n_models_to_show, len(per_model_samples))
        fig, axes = plt.subplots(rows, samples_per_model, figsize=(4 * samples_per_model, 3.5 * rows))
        if rows == 1:
            axes = np.array([axes])

        for model_idx in range(rows):
            model_samples = per_model_samples[model_idx]
            if len(model_samples) == 0:
                continue
            picks = np.random.choice(len(model_samples), size=min(samples_per_model, len(model_samples)), replace=False)
            for col, sample_idx in enumerate(picks):
                dataset.visualize(model_samples[sample_idx], ax=axes[model_idx, col])
                axes[model_idx, col].set_title(f"Model {model_idx} | sample {sample_idx}")

        fig.suptitle(title)
        plt.tight_layout()
        path = output.get_path(out_filename)
        plt.savefig(path, format='png', bbox_inches='tight', dpi=120)
        print(f"Saved sample-grid visualization to: {path}")
        plt.close(fig)

    return _viz


def _gaussian_summary_viz(dataset: GaussianMixtureDataset, title: str,
                          out_filename: str = 'gaussian_summary.pdf') -> Callable:
    def _viz(output, x_train, baseline_samples, final_samples, per_model_samples, weights):
        if baseline_samples is None:
            print("Skipping gaussian summary visualization because baseline samples are unavailable.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        x_train_2d = dataset.binary_to_continuous(x_train)
        baseline_2d = dataset.binary_to_continuous(baseline_samples)

        ax.scatter(x_train_2d[:, 0], x_train_2d[:, 1],
                   alpha=0.4, s=80, c='black', marker='x', linewidths=1.5,
                   label='Ground Truth', zorder=2)

        bmx, bmy = np.mean(baseline_2d[:, 0]), np.mean(baseline_2d[:, 1])
        bsx, bsy = np.std(baseline_2d[:, 0]), np.std(baseline_2d[:, 1])
        ax.errorbar(bmx, bmy, xerr=bsx, yerr=bsy,
                    color='gray', alpha=0.7, linewidth=2, capsize=6, zorder=3)
        ax.scatter(bmx, bmy, s=320, c='gray', marker='*',
                   edgecolors='black', linewidth=1.3, label='Baseline mean +/- std', zorder=4)

        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(per_model_samples))))
        for i, model_samples in enumerate(per_model_samples):
            if len(model_samples) == 0:
                continue
            model_2d = dataset.binary_to_continuous(model_samples)
            mx, my = np.mean(model_2d[:, 0]), np.mean(model_2d[:, 1])
            sx, sy = np.std(model_2d[:, 0]), np.std(model_2d[:, 1])
            ax.errorbar(mx, my, xerr=sx, yerr=sy,
                        color=colors[i], alpha=0.5, linewidth=1.8, capsize=4, zorder=3)
            ax.scatter(mx, my, s=180, c=[colors[i]], marker='o',
                       edgecolors='black', linewidth=1.0, label=f'Model {i}', zorder=4)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        plt.tight_layout()
        path = output.get_path(out_filename)
        plt.savefig(path, format='pdf', bbox_inches='tight')
        print(f"Saved gaussian summary visualization to: {path}")
        plt.close(fig)

    return _viz


def _ising_lorenz_viz(dataset, title: str, out_filename: str) -> Callable:
    """Lorenz / CDF visualization for the Ising Boltzmann distribution.

    Left panel:  cumulative probability mass vs number of modes covered.
    Right panel: sorted probability spectrum (exact vs empirical).
    Individual ensemble models are shown as thin coloured lines.
    """
    def _samples_to_empirical(samples, n_states, n_qubits):
        """Convert binary samples to sorted empirical probabilities and CDF."""
        indices = np.sum(
            samples.astype(int) * (2 ** np.arange(n_qubits)), axis=1,
        )
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

    def _viz(output, x_train, baseline_samples, final_samples, per_model_samples, weights):
        exact_probs = np.asarray(dataset.probs, dtype=np.float64)
        n_states = len(exact_probs)
        n_qubits = dataset.n_qubits
        p_sorted = np.sort(exact_probs)[::-1]
        cdf_exact = np.cumsum(p_sorted)
        k = np.arange(1, n_states + 1)
        pr_exact = _participation_ratio(exact_probs)
        pr_uniform = float(n_states)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Panel 1: Lorenz / CDF ---
        ax = axes[0]
        ax.plot(k, cdf_exact, label='Exact Boltzmann', color='black', linewidth=2)

        # Per-model empirical CDFs
        model_colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(per_model_samples))))
        for i, m_samples in enumerate(per_model_samples):
            if m_samples is None or len(m_samples) == 0:
                continue
            m_sorted, m_cdf = _samples_to_empirical(m_samples, n_states, n_qubits)
            w_label = f'  (w={weights[i]:.2f})' if weights is not None and i < len(weights) else ''
            ax.plot(k, m_cdf, color=model_colors[i], linewidth=0.8, alpha=0.5,
                    label=f'Model {i}{w_label}')

        # Ensemble empirical CDF
        if final_samples is not None and len(final_samples) > 0:
            emp_sorted, cdf_emp = _samples_to_empirical(final_samples, n_states, n_qubits)
            ax.plot(k, cdf_emp, label='Ensemble', color='green',
                    linewidth=2, alpha=0.9)
            pr_ensemble = _participation_ratio(emp_sorted)
        else:
            pr_ensemble = float('nan')

        # Baseline empirical CDF
        base_sorted = None
        if baseline_samples is not None and len(baseline_samples) > 0:
            base_sorted, cdf_emp_b = _samples_to_empirical(baseline_samples, n_states, n_qubits)
            ax.plot(k, cdf_emp_b, label='Baseline', color='gray',
                    linewidth=1.5, alpha=0.6, linestyle='--')
            pr_baseline = _participation_ratio(base_sorted)
        else:
            pr_baseline = float('nan')

        # Mark 90% / 99% thresholds on exact curve
        for threshold in [0.90, 0.99]:
            idx = int(np.searchsorted(cdf_exact, threshold))
            if idx < n_states:
                ax.plot(idx + 1, cdf_exact[idx], 'o', markersize=5, color='red')
                ax.annotate(f'k={idx+1}', (idx + 1, cdf_exact[idx]),
                            textcoords='offset points', xytext=(5, -12),
                            fontsize=8, color='red')

        ax.set_xlabel('Number of modes covered (k)')
        ax.set_ylabel('Cumulative probability mass')
        ax.set_title('Lorenz curve')
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, n_states)
        ax.set_ylim(0, 1.05)

        # --- Panel 2: probability spectrum ---
        ax2 = axes[1]
        ax2.plot(k, p_sorted, label='Exact Boltzmann', color='black', linewidth=1.5)
        ax2.axhline(1.0 / n_states, color='gray', linestyle='--', alpha=0.5, label='Uniform')

        # Per-model spectra
        for i, m_samples in enumerate(per_model_samples):
            if m_samples is None or len(m_samples) == 0:
                continue
            m_sorted, _ = _samples_to_empirical(m_samples, n_states, n_qubits)
            ax2.plot(k, m_sorted, color=model_colors[i], linewidth=0.6, alpha=0.4)

        if final_samples is not None and len(final_samples) > 0:
            ax2.plot(k, emp_sorted, label='Ensemble', color='green',
                     linewidth=1.5, alpha=0.8)

        if base_sorted is not None:
            ax2.plot(k, base_sorted, label='Baseline', color='gray',
                     linewidth=1.5, alpha=0.7, linestyle='--')

        # Print PR summary to suite.log and add it to the plot for quick inspection.
        print(
            "Participation ratio (effective states): "
            f"exact={pr_exact:.2f}, ensemble={pr_ensemble:.2f}, "
            f"baseline={pr_baseline:.2f}, uniform={pr_uniform:.0f}"
        )
        ax2.text(
            0.03,
            0.03,
            "\n".join([
                f"PR exact: {pr_exact:.1f}",
                f"PR ensemble: {pr_ensemble:.1f}",
                f"PR baseline: {pr_baseline:.1f}",
                f"PR uniform: {pr_uniform:.0f}",
            ]),
            transform=ax2.transAxes,
            fontsize=8,
            va='bottom',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.65, 'edgecolor': 'gray'},
        )

        ax2.set_xlabel('State rank')
        ax2.set_ylabel('Probability')
        ax2.set_title('Probability spectrum')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, n_states)

        beta = getattr(dataset, 'beta', '?')
        fig.suptitle(f'{title}  (beta={beta})', fontsize=12)
        plt.tight_layout()
        path = output.get_path(out_filename)
        plt.savefig(path, format='pdf', bbox_inches='tight')
        plt.savefig(str(path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved Ising Lorenz visualization to: {path}")
        plt.close(fig)

    return _viz


def _support_lorenz_viz(dataset, title: str, out_filename: str) -> Callable:
    """Lorenz / spectrum visualization for discrete datasets with finite support.

    Reference distribution defaults to empirical training distribution in the
    full 2^n state space. If unavailable, it falls back to uniform over
    dataset valid support.
    """

    def _samples_to_empirical(samples, n_states, n_qubits):
        if samples is None or len(samples) == 0:
            probs = np.zeros(n_states, dtype=np.float64)
            return probs, np.cumsum(np.sort(probs)[::-1])
        indices = np.sum(samples.astype(int) * (2 ** np.arange(n_qubits)), axis=1)
        counts = np.bincount(indices, minlength=n_states)
        probs = counts.astype(np.float64) / float(max(1, counts.sum()))
        sorted_p = np.sort(probs)[::-1]
        return sorted_p, np.cumsum(sorted_p)

    def _samples_to_empirical_labels(samples, label_to_idx, n_states, invalid_idx):
        counts = np.zeros(n_states, dtype=np.float64)
        if samples is None or len(samples) == 0:
            return counts, np.cumsum(np.sort(counts)[::-1])

        for s in samples:
            t = tuple(np.asarray(s, dtype=int))
            lbl = dataset.pattern_to_label.get(t)
            idx = label_to_idx.get(lbl, invalid_idx)
            counts[idx] += 1.0

        probs = counts / float(max(1.0, counts.sum()))
        sorted_p = np.sort(probs)[::-1]
        return sorted_p, np.cumsum(sorted_p)

    def _participation_ratio(probs: np.ndarray) -> float:
        probs = np.asarray(probs, dtype=np.float64)
        denom = float(np.sum(probs ** 2))
        if denom <= 0.0:
            return float('nan')
        return float(1.0 / denom)

    def _support_to_exact_probs(n_qubits: int) -> np.ndarray:
        support = getattr(dataset, '_valid_patterns', None)
        if not support:
            raise ValueError("Dataset does not expose _valid_patterns required for support Lorenz visualization.")
        n_states = 2 ** n_qubits
        probs = np.zeros(n_states, dtype=np.float64)
        support_size = len(support)
        if support_size == 0:
            return probs
        mass = 1.0 / float(support_size)
        for pat in support:
            arr = np.asarray(pat, dtype=int)
            idx = int(np.sum(arr * (2 ** np.arange(n_qubits))))
            probs[idx] = mass
        return probs

    def _training_to_exact_probs(samples: np.ndarray, n_qubits: int) -> np.ndarray:
        n_states = 2 ** n_qubits
        if samples is None or len(samples) == 0:
            return np.zeros(n_states, dtype=np.float64)
        indices = np.sum(samples.astype(int) * (2 ** np.arange(n_qubits)), axis=1)
        counts = np.bincount(indices, minlength=n_states)
        probs = counts.astype(np.float64)
        total = float(probs.sum())
        if total <= 0:
            return np.zeros(n_states, dtype=np.float64)
        return probs / total

    def _training_to_label_probs(samples: np.ndarray, label_to_idx, n_states, invalid_idx) -> np.ndarray:
        counts = np.zeros(n_states, dtype=np.float64)
        if samples is None or len(samples) == 0:
            return counts

        for s in samples:
            t = tuple(np.asarray(s, dtype=int))
            lbl = dataset.pattern_to_label.get(t)
            idx = label_to_idx.get(lbl, invalid_idx)
            counts[idx] += 1.0

        total = float(counts.sum())
        if total <= 0.0:
            return counts
        return counts / total

    def _viz(output, x_train, baseline_samples, final_samples, per_model_samples, weights):
        n_qubits = dataset.n_qubits
        use_label_space = False

        if use_label_space:
            labels = sorted(list(dataset.unique_labels))
            label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
            invalid_idx = len(labels)
            n_states = len(labels) + 1
            exact_probs = _training_to_label_probs(x_train, label_to_idx, n_states, invalid_idx)
            reference_label = 'Reference training-labels'
            x_label = 'Number of label modes covered (k)'
            rank_label = 'Label-mode rank'
            uniform_label = 'Uniform labels+invalid'
        else:
            exact_probs = _training_to_exact_probs(x_train, n_qubits)
            reference_label = 'Reference training-empirical'
            if float(np.sum(exact_probs)) <= 0.0:
                exact_probs = _support_to_exact_probs(n_qubits)
                reference_label = 'Reference support-uniform (fallback)'
            x_label = 'Number of modes covered (k)'
            rank_label = 'State rank'
            uniform_label = 'Uniform all states'

        n_states = len(exact_probs)
        p_sorted = np.sort(exact_probs)[::-1]
        cdf_exact = np.cumsum(p_sorted)
        k = np.arange(1, n_states + 1)
        pr_exact = _participation_ratio(exact_probs)
        pr_uniform = float(n_states)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(k, cdf_exact, label=reference_label, color='black', linewidth=2)

        model_colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(per_model_samples))))
        for i, m_samples in enumerate(per_model_samples):
            if m_samples is None or len(m_samples) == 0:
                continue
            if use_label_space:
                _, m_cdf = _samples_to_empirical_labels(m_samples, label_to_idx, n_states, invalid_idx)
            else:
                _, m_cdf = _samples_to_empirical(m_samples, n_states, n_qubits)
            w_label = f'  (w={weights[i]:.2f})' if weights is not None and i < len(weights) else ''
            ax.plot(k, m_cdf, color=model_colors[i], linewidth=0.8, alpha=0.5,
                    label=f'Model {i}{w_label}')

        if final_samples is not None and len(final_samples) > 0:
            if use_label_space:
                emp_sorted, cdf_emp = _samples_to_empirical_labels(final_samples, label_to_idx, n_states, invalid_idx)
            else:
                emp_sorted, cdf_emp = _samples_to_empirical(final_samples, n_states, n_qubits)
            ax.plot(k, cdf_emp, label='Ensemble', color='green', linewidth=2, alpha=0.9)
            pr_ensemble = _participation_ratio(emp_sorted)
        else:
            pr_ensemble = float('nan')

        base_sorted = None
        if baseline_samples is not None and len(baseline_samples) > 0:
            if use_label_space:
                base_sorted, cdf_emp_b = _samples_to_empirical_labels(baseline_samples, label_to_idx, n_states, invalid_idx)
            else:
                base_sorted, cdf_emp_b = _samples_to_empirical(baseline_samples, n_states, n_qubits)
            ax.plot(k, cdf_emp_b, label='Baseline', color='gray', linewidth=1.5, alpha=0.6, linestyle='--')
            pr_baseline = _participation_ratio(base_sorted)
        else:
            pr_baseline = float('nan')

        for threshold in [0.90, 0.99]:
            idx = int(np.searchsorted(cdf_exact, threshold))
            if idx < n_states:
                ax.plot(idx + 1, cdf_exact[idx], 'o', markersize=5, color='red')
                ax.annotate(f'k={idx+1}', (idx + 1, cdf_exact[idx]),
                            textcoords='offset points', xytext=(5, -12), fontsize=8, color='red')

        ax.set_xlabel(x_label)
        ax.set_ylabel('Cumulative probability mass')
        ax.set_title('Lorenz curve')
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, n_states)
        ax.set_ylim(0, 1.05)

        ax2 = axes[1]
        ax2.plot(k, p_sorted, label=reference_label, color='black', linewidth=1.5)
        ax2.axhline(1.0 / n_states, color='gray', linestyle='--', alpha=0.5, label=uniform_label)

        for i, m_samples in enumerate(per_model_samples):
            if m_samples is None or len(m_samples) == 0:
                continue
            if use_label_space:
                m_sorted, _ = _samples_to_empirical_labels(m_samples, label_to_idx, n_states, invalid_idx)
            else:
                m_sorted, _ = _samples_to_empirical(m_samples, n_states, n_qubits)
            ax2.plot(k, m_sorted, color=model_colors[i], linewidth=0.6, alpha=0.4)

        if final_samples is not None and len(final_samples) > 0:
            ax2.plot(k, emp_sorted, label='Ensemble', color='green', linewidth=1.5, alpha=0.8)

        if base_sorted is not None:
            ax2.plot(k, base_sorted, label='Baseline', color='gray', linewidth=1.5, alpha=0.7, linestyle='--')

        print(
            "Participation ratio (effective states): "
            f"exact={pr_exact:.2f}, ensemble={pr_ensemble:.2f}, "
            f"baseline={pr_baseline:.2f}, uniform={pr_uniform:.0f}"
        )
        ax2.text(
            0.03,
            0.03,
            "\n".join([
                f"PR exact: {pr_exact:.1f}",
                f"PR ensemble: {pr_ensemble:.1f}",
                f"PR baseline: {pr_baseline:.1f}",
                f"PR uniform: {pr_uniform:.0f}",
            ]),
            transform=ax2.transAxes,
            fontsize=8,
            va='bottom',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.65, 'edgecolor': 'gray'},
        )

        ax2.set_xlabel(rank_label)
        ax2.set_ylabel('Probability')
        ax2.set_title('Probability spectrum')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, n_states)

        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        path = output.get_path(out_filename)
        plt.savefig(path, format='pdf', bbox_inches='tight')
        plt.savefig(str(path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved support Lorenz visualization to: {path}")
        plt.close(fig)

    return _viz


def _resolve_plot_kind(dataset_key: str, plot_spec: dict | None) -> str:
    if plot_spec and 'kind' in plot_spec:
        return str(plot_spec['kind']).lower()

    defaults = {
        'bas': 'lorenz',
        'parity': 'histogram',
        'blobs': 'lorenz',
        'shapes': 'sample_grid',
        'gaussian': 'gaussian_summary',
        'ising': 'ising_lorenz',
        'mnist': 'sample_grid',
        'dwave': 'histogram',
        'scale_free': 'histogram',
        'genomic': 'histogram',
        'noisy_bas': 'lorenz',
        'random_circuit': 'lorenz',
        'qaoa_maxcut': 'lorenz',
        'tfim_thermal': 'lorenz',
        'rydberg': 'lorenz',
    }
    return defaults.get(dataset_key, 'none')


def _build_custom_viz(dataset_key: str, dataset_obj, plot_spec: dict | None, n_qubits: int) -> Callable | None:
    plot_spec = plot_spec or {}
    kind = _resolve_plot_kind(dataset_key, plot_spec)
    params = plot_spec.get('params', {}) if isinstance(plot_spec, dict) else {}

    if kind == 'none':
        return None
    if kind == 'histogram':
        title = params.get('title', f'{dataset_key.upper()} distribution comparison')
        filename = params.get('filename', f'{dataset_key}_distribution.pdf')
        return _histogram_viz(title=title, out_filename=filename, n_qubits=n_qubits)
    if kind == 'sample_grid':
        title = params.get('title', f'{dataset_key.upper()} per-model samples')
        filename = params.get('filename', f'{dataset_key}_samples.png')
        n_models_to_show = int(params.get('n_models_to_show', 3))
        samples_per_model = int(params.get('samples_per_model', 4))
        return _sample_grid_viz(
            dataset=dataset_obj,
            title=title,
            out_filename=filename,
            n_models_to_show=n_models_to_show,
            samples_per_model=samples_per_model,
        )
    if kind == 'gaussian_summary':
        title = params.get('title', 'Gaussian mixture: baseline and model summaries')
        filename = params.get('filename', 'gaussian_summary.pdf')
        return _gaussian_summary_viz(dataset_obj, title=title, out_filename=filename)
    if kind == 'ising_lorenz':
        title = params.get('title', 'Frustrated Ising distribution')
        filename = params.get('filename', 'ising_lorenz.pdf')
        return _ising_lorenz_viz(dataset_obj, title=title, out_filename=filename)
    if kind == 'lorenz':
        title = params.get('title', f'{dataset_key.upper()} distribution')
        filename = params.get('filename', f'{dataset_key}_lorenz.pdf')
        if hasattr(dataset_obj, 'probs'):
            return _ising_lorenz_viz(dataset_obj, title=title, out_filename=filename)
        return _support_lorenz_viz(dataset_obj, title=title, out_filename=filename)

    raise ValueError(f"Unknown plot kind '{kind}'.")


def build_dataset_bundle(dataset_spec: dict, config: dict, plot_spec: dict | None = None) -> dict:
    """Create dataset, generated training set, metric callables, and optional viz callback."""
    if not isinstance(dataset_spec, dict):
        raise ValueError("Each run requires a 'dataset' object with at least a 'name'.")

    dataset_key = str(dataset_spec.get('name', '')).strip().lower()
    params = dataset_spec.get('params', {})

    train_samples = int(config.get('train_samples', 1000))
    data_seed = int(config.get('data_seed', 0))

    if dataset_key == 'bas':
        height, width = _resolve_rows_cols(params, config, default=(4, 4))
        ds = BarsAndStripesDataset(height=height, width=width)
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'BAS ({height}x{width})'
        n_qubits = height * width

    elif dataset_key == 'noisy_bas':
        height, width = _resolve_rows_cols(params, config, default=(4, 4))
        flip_prob = float(params.get('flip_prob', config.get('flip_prob', 0.05)))
        ds = NoisyBASDataset(height=height, width=width, flip_prob=flip_prob)
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'Noisy BAS ({height}x{width}, p={flip_prob:g})'
        n_qubits = height * width

    elif dataset_key == 'parity':
        n_qubits = int(params.get('n_qubits', config.get('dim', config.get('n_qubits', 6))))
        ds = ParityDataset(n_qubits=n_qubits)
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'Parity ({n_qubits} qubits)'

    elif dataset_key == 'blobs':
        ds = BlobsDataset()
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = 'Blobs'
        n_qubits = x_train.shape[1]

    elif dataset_key == 'gaussian':
        grid_size = int(params.get('grid_size', config.get('dims', (4, 4))[0]))
        spread = float(params.get('spread', config.get('gaussian_spread', 0.3)))
        separation = float(params.get('separation', config.get('gaussian_separation', 3.0)))
        grid_bits = int(params.get('grid_bits', config.get('grid_bits', 4)))
        ds = GaussianMixtureDataset(
            grid_size=grid_size,
            spread=spread,
            separation=separation,
            grid_bits=grid_bits,
        )
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'Gaussian Mixture ({grid_size}x{grid_size})'
        n_qubits = x_train.shape[1]

    elif dataset_key == 'shapes':
        grid_shape = tuple(params.get('grid_shape', config.get('grid_shape', (5, 5))))
        shape_types = params.get('shape_types', None)
        ds = BinaryShapesDataset(grid_shape=grid_shape, shape_types=shape_types)
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'Binary Shapes ({grid_shape[0]}x{grid_shape[1]})'
        n_qubits = x_train.shape[1]

    elif dataset_key == 'ising':
        rows = int(params.get('rows', 4))
        cols = int(params.get('cols', 4))
        beta = float(params.get('beta', 2.0))
        j_seed = int(params.get('j_seed', 0))
        ds = FrustratedIsingDataset(
            rows=rows, cols=cols, beta=beta, j_seed=j_seed,
        )
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'Frustrated Ising ({rows}x{cols}, beta={beta})'
        n_qubits = x_train.shape[1]

    elif dataset_key == 'pennylane_ising':
        rows, cols = _resolve_rows_cols(params, config, default=(4, 4))
        ds = PennylaneIsingDataset(rows=rows, cols=cols)
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'PennyLane Ising ({rows}x{cols})'
        n_qubits = x_train.shape[1]

    elif dataset_key == 'pennylane_bas':
        rows, cols = _resolve_rows_cols(params, config, default=(4, 4))
        ds = PennylaneBASDataset(rows=rows, cols=cols)
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'PennyLane BAS ({rows}x{cols})'
        n_qubits = x_train.shape[1]

    elif dataset_key == 'pennylane_hm':
        rows, cols = _resolve_rows_cols(params, config, default=(4, 4))
        ds = PennylaneHMDataset(rows=rows, cols=cols)
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'PennyLane HM ({rows}x{cols})'
        n_qubits = x_train.shape[1]

    elif dataset_key == 'mnist':
        rows, cols = _resolve_rows_cols(params, config, default=(4, 4))
        digit = params.get('digit', None)
        if digit is not None:
            digit = int(digit)
        reduction = str(params.get('reduction', 'spatial'))
        ds = BinarizedMNISTDataset(
            rows=rows, cols=cols, digit=digit, reduction=reduction,
        )
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        digit_str = f', digit={digit}' if digit is not None else ''
        dataset_name = f'Binarized MNIST ({rows}x{cols}{digit_str})'
        n_qubits = x_train.shape[1]

    elif dataset_key == 'dwave':
        rows, cols = _resolve_rows_cols(params, config, default=(4, 4))
        ds = DWaveDataset(rows=rows, cols=cols)
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'D-Wave ({rows}x{cols})'
        n_qubits = x_train.shape[1]

    elif dataset_key == 'scale_free':
        rows, cols = _resolve_rows_cols(params, config, default=(4, 4))
        ds = ScaleFreeDataset(rows=rows, cols=cols)
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'Scale-Free ({rows}x{cols})'
        n_qubits = x_train.shape[1]

    elif dataset_key == 'genomic':
        rows, cols = _resolve_rows_cols(params, config, default=(4, 4))
        ds = GenomicDataset(rows=rows, cols=cols)
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'Genomic ({rows}x{cols})'
        n_qubits = x_train.shape[1]

    elif dataset_key == 'random_circuit':
        n_qubits = int(params.get('n_qubits', 12))
        depth = params.get('depth', None)
        if depth is not None:
            depth = int(depth)
        circuit_seed = int(params.get('circuit_seed', 42))
        ds = RandomCircuitDataset(
            n_qubits=n_qubits, depth=depth, circuit_seed=circuit_seed,
        )
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'Random Circuit (N={n_qubits}, d={ds.depth})'

    elif dataset_key == 'qaoa_maxcut':
        n_qubits = int(params.get('n_qubits', 12))
        graph_type = str(params.get('graph_type', 'random_regular'))
        graph_seed = int(params.get('graph_seed', 42))
        p_depth = int(params.get('p_depth', 1))
        param_seed = int(params.get('param_seed', 42))
        ds = QAOAMaxCutDataset(
            n_qubits=n_qubits, graph_type=graph_type,
            graph_seed=graph_seed, p_depth=p_depth, param_seed=param_seed,
        )
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'QAOA MaxCut (N={n_qubits}, p={p_depth})'

    elif dataset_key == 'tfim_thermal':
        n_qubits = int(params.get('n_qubits', 12))
        temperature = float(params.get('temperature', 1.0))
        coupling = float(params.get('coupling', 1.0))
        h_field = float(params.get('h_field', 1.0))
        boundary_condition = bool(params.get('boundary_condition', False))
        ds = TFIMThermalDataset(
            n_qubits=n_qubits, temperature=temperature,
            coupling=coupling, h_field=h_field,
            boundary_condition=boundary_condition,
        )
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'TFIM Thermal (N={n_qubits}, T={temperature})'

    elif dataset_key == 'rydberg':
        n_qubits = int(params.get('n_qubits', 16))
        dataset_index = int(params.get('dataset_index', 0))
        ds = RydbergDataset(
            n_qubits=n_qubits, dataset_index=dataset_index,
        )
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'Rydberg (N={n_qubits})'

    else:
        raise ValueError(
            "Unknown dataset name. Supported values: "
            "bas, noisy_bas, blobs, dwave, gaussian, genomic, ising, mnist, "
            "parity, qaoa_maxcut, random_circuit, rydberg, scale_free, shapes, "
            "tfim_thermal."
        )

    custom_viz_fn = _build_custom_viz(dataset_key, ds, plot_spec, n_qubits)

    # Datasets where validity/coverage are not meaningful pass None
    # so evaluate_samples() skips those metrics (reports NaN).
    # Datasets without a meaningful pattern space pass None for validity/coverage.
    no_pattern_space = {'ising', 'noisy_bas', 'mnist', 'dwave', 'scale_free', 'genomic', 'pennylane_ising', 'pennylane_bas', 'pennylane_hm', 'random_circuit', 'qaoa_maxcut', 'tfim_thermal', 'rydberg'}
    if dataset_key in no_pattern_space:
        validity_fn = None
        coverage_fn = None
        top_k_tvd_fn = getattr(ds, 'top_k_tvd', None)
    else:
        validity_fn = ds.validity_rate
        coverage_fn = ds.coverage_rate
        top_k_tvd_fn = None

    return {
        'dataset_name': dataset_name,
        'x_train': x_train,
        'validity_fn': validity_fn,
        'coverage_fn': coverage_fn,
        'top_k_tvd_fn': top_k_tvd_fn,
        'custom_viz_fn': custom_viz_fn,
    }
