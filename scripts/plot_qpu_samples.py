"""Plot QPU samples from an ibm-q ensemble_mixture run with per-model labels and ground truth comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.mnist import MNISTDataset
from src.core.metrics import (
    compute_mmd,
    compute_tvd,
    compute_kl_divergence,
    compute_jsd,
    compute_precision_recall_f1,
)


def load_qpu_samples(data_dir: Path) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Load per-model samples + combined mixture + metadata from ensemble_mixture output.

    Returns:
        per_model: list of (n_shots, 100) int8 arrays, one per model
        combined: (total_shots, 100) int8 array concatenating all models
        metadata: dict from metadata.json
    """
    samples_npz = data_dir / "samples.npz"
    metadata_path = data_dir / "metadata.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    with np.load(samples_npz) as d:
        blocks = [d[f"block_{i}"] for i in range(len(d))]

    per_model = []
    for i, block in enumerate(blocks):
        arr = np.array([[int(c) for c in s] for s in block], dtype=np.int8)
        per_model.append(arr)

    combined = np.vstack(per_model) if per_model else np.empty((0, 100), dtype=np.int8)
    return per_model, combined, metadata


def generate_ground_truth(n_samples: int = 50, seed: int = 42) -> np.ndarray:
    ds = MNISTDataset(rows=10, cols=10, threshold=0.4, classes=[0])
    return ds.generate(n_samples, seed=seed, split="train")


def plot_qpu_samples_per_model(per_model: list[np.ndarray], weights: list[float],
                               save_dir: Path) -> plt.Figure:
    """Grid: 2 rows × 5 cols, each cell shows 3 samples from one model + shot count."""
    n_models = len(per_model)
    samples_per_cell = 3
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), constrained_layout=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_models:
            ax.axis("off")
            continue
        pool = per_model[i]
        n_shots = len(pool)
        shown = min(samples_per_cell, n_shots)
        idxs = np.random.choice(n_shots, size=shown, replace=False)
        for j in range(shown):
            sub = ax.inset_axes([j * 0.35, 0.05, 0.3, 0.9])
            sub.imshow(pool[idxs[j]].reshape(10, 10), cmap="gray_r", interpolation="nearest")
            sub.axis("off")
        ax.set_title(f"M{i}  ({n_shots} shots, {weights[i]:.2%})", fontsize=9)
        ax.axis("off")

    fig.suptitle("IBM Marrakesh QPU Samples — Per-Model (FCFW-weighted shots)",
                 fontsize=13, y=1.02)
    path = save_dir / "qpu_per_model.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_side_by_side(per_model: list[np.ndarray], weights: list[float],
                      ground_truth: np.ndarray, metadata: dict,
                      save_dir: Path) -> plt.Figure:
    """Side-by-side comparison: QPU mixture samples vs ground truth MNIST-0."""
    n_cols = 8

    mixture: list[np.ndarray] = []
    w = np.asarray(weights, dtype=float)
    counts = np.maximum(1, np.round(w * 100).astype(int))
    for pool, c in zip(per_model, counts):
        idxs = np.random.choice(len(pool), size=min(c, len(pool)), replace=False)
        mixture.append(pool[idxs])
    mixture_arr = np.vstack(mixture)

    gt_sample = ground_truth[
        np.random.choice(len(ground_truth), size=n_cols, replace=False)
    ]

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 1.6, 4), constrained_layout=True)

    for col in range(n_cols):
        ax_mix = axes[0, col]
        qpu_idx = np.random.choice(len(mixture_arr))
        ax_mix.imshow(mixture_arr[qpu_idx].reshape(10, 10), cmap="gray_r", interpolation="nearest")
        ax_mix.axis("off")
        if col == 0:
            ax_mix.set_ylabel("QPU\nMixture", fontsize=10, rotation=0, labelpad=20, va="center")

        ax_gt = axes[1, col]
        ax_gt.imshow(gt_sample[col].reshape(10, 10), cmap="gray_r", interpolation="nearest")
        ax_gt.axis("off")
        if col == 0:
            ax_gt.set_ylabel("MNIST-0\nGround Truth", fontsize=10, rotation=0, labelpad=20, va="center")

    backend = metadata.get("backend_name", "unknown")
    fig.suptitle(f"QPU ({backend}) vs Ground Truth — MNIST-0, FCFW-weighted ensemble (512 shots)",
                 fontsize=12, y=1.02)
    path = save_dir / "qpu_vs_ground_truth.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_ensemble_grid(per_model: list[np.ndarray], combined: np.ndarray,
                       weights: list[float], ground_truth: np.ndarray,
                       metadata: dict, save_dir: Path) -> plt.Figure:
    """Comprehensive figure: one row of QPU samples + ground truth summary metrics."""
    w = np.asarray(weights, dtype=float)
    gt_sample = ground_truth[
        np.random.choice(len(ground_truth), size=10, replace=False)
    ]

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[3, 1])

    # Top: QPU mixture samples
    sub_top = subfigs[0]
    rows, cols = 4, 8
    axs_top = sub_top.subplots(rows, cols)
    sub_top.suptitle("QPU Ensemble Mixture Samples (FCFW-weighted)", fontsize=13)

    mixture_samples = []
    pool_indices = []
    for mi, (pool, wi) in enumerate(zip(per_model, w)):
        n_from_pool = max(1, round(wi * 32))
        idxs = np.random.choice(len(pool), size=min(n_from_pool, len(pool)), replace=False)
        mixture_samples.extend(pool[idxs])
        pool_indices.extend([mi] * len(idxs))
    mixture_arr = np.array(mixture_samples)
    pool_arr = np.array(pool_indices)

    n_to_show = min(rows * cols, len(mixture_arr))
    shown_idxs = np.random.choice(len(mixture_arr), size=n_to_show, replace=False)
    cmap = plt.colormaps["tab10"]

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axs_top[r, c]
        if idx < n_to_show:
            si = shown_idxs[idx]
            ax.imshow(mixture_arr[si].reshape(10, 10), cmap="gray_r", interpolation="nearest")
            mi = pool_arr[si]
            ax.set_title(f"M{mi}", fontsize=7, color=cmap(mi % 10), pad=1)
        ax.axis("off")

    # Bottom: ground truth + stats
    sub_bot = subfigs[1]
    sub_bot.suptitle("Ground Truth MNIST-0", fontsize=13)
    axs_bot = sub_bot.subplots(1, 12)
    for i in range(12):
        ax = axs_bot[i]
        if i < 10:
            ax.imshow(gt_sample[i].reshape(10, 10), cmap="gray_r", interpolation="nearest")
        ax.axis("off")

    # Last two cells: metadata
    axs_bot[10].axis("off")
    axs_bot[11].axis("off")
    backend = metadata.get("backend_name", "unknown")
    total_shots = metadata.get("total_shots", sum(len(p) for p in per_model))
    info_text = (
        f"Backend: {backend}\n"
        f"Total shots: {total_shots}\n"
        f"Models: {len(per_model)}\n"
        f"Unique outcomes: {len(np.unique(combined, axis=0))}\n"
        f"Mean Hamming weight: {combined.sum(axis=1).mean():.1f}"
    )
    axs_bot[10].text(0, 0.5, info_text, fontsize=8, va="center",
                     transform=axs_bot[10].transAxes)
    axs_bot[10].axis("off")
    for idx in range(11, 12):
        axs_bot[idx].axis("off")

    path = save_dir / "qpu_comprehensive.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fig


def compute_qpu_metrics(per_model: list[np.ndarray], combined: np.ndarray,
                         weights: list[float], ground_truth: np.ndarray,
                         metadata: dict, save_dir: Path) -> dict:
    """Compute all metrics (MMD, TVD, KL, JSD, Precision/Recall/F1) and save to JSON."""
    sigma = metadata.get("sigma", 2.0)
    if isinstance(sigma, list):
        sigma = sigma[0]

    mixture = combine_samples_by_weights(per_model, np.asarray(weights, dtype=float), 512)

    gt_sum = ground_truth.sum()
    nq = ground_truth.shape[1]
    print(f"\nMetrics (sigma={sigma}):")
    print(f"  Ground truth: {len(ground_truth)} samples, mean HW={gt_sum / len(ground_truth):.1f}")

    results: dict = {
        "backend": metadata.get("backend_name", "unknown"),
        "timestamp": metadata.get("completed_at_utc", ""),
        "total_shots": metadata.get("total_shots", sum(len(p) for p in per_model)),
        "n_models": len(per_model),
        "weights": weights,
    }

    methods = [("Mixture (FCFW)", mixture)]
    for mi in range(len(per_model)):
        methods.append((f"Model {mi}", per_model[mi]))
    methods.append(("Ground Truth", ground_truth))

    for label, samples in methods:
        if len(samples) == 0:
            continue
        hw = samples.sum(axis=1)
        entry: dict = {
            "n_samples": int(len(samples)),
            "mean_hamming_weight": float(hw.mean()),
            "std_hamming_weight": float(hw.std()),
            "median_hamming_weight": float(np.median(hw)),
            "min_hamming_weight": int(hw.min()),
            "max_hamming_weight": int(hw.max()),
        }

        if label not in ("Ground Truth",):
            try:
                entry["mmd"] = float(compute_mmd(ground_truth, samples, sigma))
            except Exception as e:
                entry["mmd"] = float("nan")
            try:
                entry["tvd"] = float(compute_tvd(ground_truth, samples))
            except Exception as e:
                entry["tvd"] = float("nan")
            try:
                entry["kl"] = float(compute_kl_divergence(ground_truth, samples))
            except Exception as e:
                entry["kl"] = float("nan")
            try:
                entry["jsd"] = float(compute_jsd(ground_truth, samples))
            except Exception as e:
                entry["jsd"] = float("nan")
            try:
                pr = compute_precision_recall_f1(ground_truth, samples, sigma)
                entry.update(pr)
            except Exception as e:
                entry["precision"] = float("nan")
                entry["recall"] = float("nan")
                entry["f_score"] = float("nan")

        results[label] = entry

        if label.startswith("Mixture") or label.startswith("Model"):
            print(f"  {label:20s}: {len(samples):4d} shots, "
                  f"HW={hw.mean():.1f}±{hw.std():.1f}, "
                  f"MMD={entry.get('mmd', float('nan')):.6f}, "
                  f"TVD={entry.get('tvd', float('nan')):.4f}")

    path = save_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved metrics: {path}")
    return results


def combine_samples_by_weights(per_model: list[np.ndarray],
                                weights: np.ndarray, total_n: int) -> np.ndarray:
    """Draw samples from per-model pools proportional to weights."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(per_model), dtype=float)
    w = w / w.sum()
    counts = np.maximum(0, np.round(w * total_n).astype(int))
    remainder = total_n - counts.sum()
    if remainder > 0:
        idxs = np.argsort(-w)
        for i in range(remainder):
            counts[idxs[i % len(idxs)]] += 1
    selected = []
    for pool, c in zip(per_model, counts):
        if c <= 0 or len(pool) == 0:
            continue
        if c <= len(pool):
            idxs = np.random.choice(len(pool), size=c, replace=False)
        else:
            idxs = np.random.choice(len(pool), size=c, replace=True)
        selected.append(pool[idxs])
    if not selected:
        return np.empty((0, per_model[0].shape[1]), dtype=np.int8)
    return np.vstack(selected)


def plot_hamming_weight_distribution(per_model: list[np.ndarray],
                                      combined: np.ndarray,
                                      weights: list[float],
                                      ground_truth: np.ndarray,
                                      metadata: dict,
                                      save_dir: Path) -> plt.Figure:
    """Histogram: Hamming weight distribution of QPU mixture vs ground truth, with optional per-model inset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    mixture = combine_samples_by_weights(per_model, np.asarray(weights, dtype=float), 512)
    gt_hw = ground_truth.sum(axis=1)
    mixture_hw = mixture.sum(axis=1)
    combined_hw = combined.sum(axis=1)
    max_hw = max(gt_hw.max(), combined_hw.max())

    bins = np.arange(0, max_hw + 2) - 0.5

    axes[0].hist(gt_hw, bins=bins, density=True, alpha=0.5, color="gray",
                 edgecolor="gray", linewidth=0.5, label=f"Ground Truth (n={len(gt_hw)})")
    axes[0].hist(mixture_hw, bins=bins, density=True, alpha=0.6, color="tab:green",
                 edgecolor="tab:green", linewidth=0.5, label=f"QPU Mixture (n={len(mixture_hw)})")
    axes[0].axvline(gt_hw.mean(), color="gray", ls="--", lw=1.2,
                    label=f"GT mean={gt_hw.mean():.1f}")
    axes[0].axvline(mixture_hw.mean(), color="tab:green", ls="--", lw=1.2,
                    label=f"QPU mean={mixture_hw.mean():.1f}")
    axes[0].set_xlabel("Hamming weight (black pixels)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Hamming Weight Distribution")
    axes[0].legend(fontsize=8)

    # Per-model individual distributions
    cmap = plt.colormaps["tab10"]
    for mi in range(len(per_model)):
        hw = per_model[mi].sum(axis=1)
        axes[1].hist(hw, bins=bins, density=True, alpha=0.35, color=cmap(mi % 10),
                     edgecolor=cmap(mi % 10), linewidth=0.3, label=f"M{mi} ({len(hw)} shots)")
    axes[1].hist(gt_hw, bins=bins, density=True, alpha=0.8, color="black",
                 histtype="step", linewidth=1.5, label=f"Ground Truth")
    axes[1].set_xlabel("Hamming weight (black pixels)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Per-Model Hamming Weight Distribution")
    axes[1].legend(fontsize=7, ncol=2)

    path = save_dir / "qpu_hamming_weight.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot QPU samples with per-model labels and ground truth comparison."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to ensemble_mixture directory (containing samples.npz, metadata.json, counts.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: data_dir/plots).",
    )
    parser.add_argument(
        "--ground-truth-seed",
        type=int,
        default=42,
        help="Seed for MNIST-0 ground truth sampling.",
    )
    parser.add_argument(
        "--ground-truth-count",
        type=int,
        default=50,
        help="Number of ground truth samples to generate.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_model, combined, metadata = load_qpu_samples(data_dir)
    fcfw_weights = metadata.get("ensemble_weights", [])
    backend = metadata.get("backend_name", "unknown")

    print(f"Loaded {len(per_model)} models from {backend}")
    for i, p in enumerate(per_model):
        print(f"  Model {i}: {len(p)} shots")
    print(f"  Total shots: {sum(len(p) for p in per_model)}")
    print(f"  Unique outcomes: {len(np.unique(combined, axis=0))}")
    print(f"  Mean Hamming weight: {combined.sum(axis=1).mean():.1f}")

    ground_truth = generate_ground_truth(args.ground_truth_count, args.ground_truth_seed)
    print(f"Generated {len(ground_truth)} ground truth MNIST-0 samples")
    print(f"  Mean Hamming weight: {ground_truth.sum() / len(ground_truth):.1f}")

    compute_qpu_metrics(per_model, combined, fcfw_weights, ground_truth, metadata, output_dir)

    plot_qpu_samples_per_model(per_model, fcfw_weights, output_dir)
    print(f"  Saved: {output_dir / 'qpu_per_model.png'}")

    plot_side_by_side(per_model, fcfw_weights, ground_truth, metadata, output_dir)
    print(f"  Saved: {output_dir / 'qpu_vs_ground_truth.png'}")

    plot_ensemble_grid(per_model, combined, fcfw_weights, ground_truth, metadata, output_dir)
    print(f"  Saved: {output_dir / 'qpu_comprehensive.png'}")

    plot_hamming_weight_distribution(per_model, combined, fcfw_weights, ground_truth, metadata, output_dir)
    print(f"  Saved: {output_dir / 'qpu_hamming_weight.png'}")


if __name__ == "__main__":
    main()
