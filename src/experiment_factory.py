"""Dataset and plotting factories for config-driven experiments."""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from src.datasets.bas import BarsAndStripesDataset
from src.datasets.blobs import BlobsDataset
from src.datasets.gaussian import GaussianMixtureDataset
from src.datasets.parity import ParityDataset
from src.datasets.shapes import BinaryShapesDataset


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


def _resolve_plot_kind(dataset_key: str, plot_spec: dict | None) -> str:
    if plot_spec and 'kind' in plot_spec:
        return str(plot_spec['kind']).lower()

    defaults = {
        'bas': 'histogram',
        'parity': 'histogram',
        'blobs': 'sample_grid',
        'shapes': 'sample_grid',
        'gaussian': 'gaussian_summary',
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
        dims = params.get('dims', config.get('dims', (4, 4)))
        height, width = int(dims[0]), int(dims[1])
        ds = BarsAndStripesDataset(height=height, width=width)
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        dataset_name = f'BAS ({height}x{width})'
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

    else:
        raise ValueError(
            "Unknown dataset name. Supported values: bas, blobs, gaussian, parity, shapes."
        )

    custom_viz_fn = _build_custom_viz(dataset_key, ds, plot_spec, n_qubits)

    return {
        'dataset_name': dataset_name,
        'x_train': x_train,
        'validity_fn': ds.validity_rate,
        'coverage_fn': ds.coverage_rate,
        'custom_viz_fn': custom_viz_fn,
    }
