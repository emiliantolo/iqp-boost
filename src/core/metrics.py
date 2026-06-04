"""Sample and evaluation metrics for IQP boosting."""

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


def compute_hamming_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Computes the exact Hamming distance matrix using BLAS."""
    X_f = X.astype(np.float32)
    Y_f = Y.astype(np.float32)
    X_sum = np.sum(X_f, axis=1, keepdims=True)
    Y_sum = np.sum(Y_f, axis=1)
    return X_sum + Y_sum - 2 * (X_f @ Y_f.T)


def fast_binary_gaussian_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian kernel on binary strings: K(x,y) = exp(-H / 2sigma^2)."""
    H = compute_hamming_matrix(X, Y)
    return np.exp(-H / (2 * sigma**2))


def compute_distributions(
    ground_truth: np.ndarray,
    model_samples: np.ndarray,
    n_bins: int = None,
    smoothing: float = 1e-10,
    max_qubits: int = 20,
    exact_probs: np.ndarray = None,
    model_probs: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute probability distributions from discrete samples or exact probs.

    When ``exact_probs`` is provided (a 2^n probability vector), it is used
    directly as the ground-truth distribution instead of building a histogram
    from ``ground_truth`` samples. This gives exact, noise-free reference
    values for TVD / KL / JSD.

    When ``model_probs`` is provided, it is used directly as the model
    distribution instead of building a histogram from ``model_samples``.

    For <=max_qubits: uses full 2^n histogram.
    For >max_qubits: uses empirical support (only observed bitstrings are binned).
    """
    if model_probs is not None:
        p_model = np.asarray(model_probs, dtype=np.float64)
        p_model = p_model / p_model.sum()
        if exact_probs is not None:
            p_data = np.asarray(exact_probs, dtype=np.float64)
            p_data = p_data / p_data.sum()
        else:
            ground_truth = np.asarray(ground_truth, dtype=int)
            gt_ints = np.sum(ground_truth * (2 ** np.arange(ground_truth.shape[1])), axis=1)
            gt_counts = np.bincount(gt_ints, minlength=len(p_model)).astype(np.float64)
            if smoothing > 0:
                gt_counts += smoothing
            total = gt_counts.sum()
            p_data = gt_counts / total if total > 0 else gt_counts
        return p_data, p_model

    model_samples = np.asarray(model_samples, dtype=int)
    n_features = model_samples.shape[1]

    def binary_to_int(samples):
        return np.sum(samples * (2 ** np.arange(samples.shape[1])), axis=1)

    if exact_probs is not None:
        p_data = np.asarray(exact_probs, dtype=np.float64)
        n_bins_exact = len(p_data)
        model_ints = binary_to_int(model_samples)
        model_counts = np.bincount(model_ints, minlength=n_bins_exact).astype(np.float64)
        if smoothing > 0:
            model_counts += smoothing
        total = model_counts.sum()
        p_model = model_counts / total if total > 0 else model_counts
        return p_data, p_model

    ground_truth = np.asarray(ground_truth, dtype=int)

    if n_features > max_qubits and n_bins is None:
        def rows_to_tuples(arr):
            return [tuple(row) for row in arr]

        gt_tuples = rows_to_tuples(ground_truth)
        model_tuples = rows_to_tuples(model_samples)
        all_keys = sorted(set(gt_tuples) | set(model_tuples))
        key_to_idx = {k: i for i, k in enumerate(all_keys)}

        gt_counts = np.zeros(len(all_keys)) + smoothing
        model_counts = np.zeros(len(all_keys)) + smoothing

        for t in gt_tuples:
            gt_counts[key_to_idx[t]] += 1
        for t in model_tuples:
            model_counts[key_to_idx[t]] += 1

        p_data = gt_counts / gt_counts.sum()
        p_model = model_counts / model_counts.sum()
        return p_data, p_model

    gt_ints = binary_to_int(ground_truth)
    model_ints = binary_to_int(model_samples)

    if n_bins is None:
        n_bins = 2 ** n_features

    gt_counts = np.bincount(gt_ints, minlength=n_bins) + smoothing
    model_counts = np.bincount(model_ints, minlength=n_bins) + smoothing

    p_data = gt_counts / gt_counts.sum()
    p_model = model_counts / model_counts.sum()
    return p_data, p_model


def compute_kl_divergence(
    ground_truth: np.ndarray,
    model_samples: np.ndarray,
    n_bins: int = None,
    smoothing: float = 1e-10,
    exact_probs: np.ndarray = None,
    model_probs: np.ndarray = None,
) -> float:
    """Compute KL divergence between discrete sample distributions using Scipy."""
    p_data, p_model = compute_distributions(
        ground_truth,
        model_samples,
        n_bins,
        smoothing,
        exact_probs=exact_probs,
        model_probs=model_probs,
    )
    return float(entropy(p_data, p_model))


def compute_jsd(
    ground_truth: np.ndarray,
    model_samples: np.ndarray,
    n_bins: int = None,
    smoothing: float = 1e-10,
    exact_probs: np.ndarray = None,
    model_probs: np.ndarray = None,
) -> float:
    """Compute Jensen-Shannon Distance (metric) using Scipy."""
    p_data, p_model = compute_distributions(
        ground_truth,
        model_samples,
        n_bins,
        smoothing,
        exact_probs=exact_probs,
        model_probs=model_probs,
    )
    return float(jensenshannon(p_data, p_model))


def compute_tvd(
    ground_truth: np.ndarray,
    model_samples: np.ndarray,
    n_bins: int = None,
    exact_probs: np.ndarray = None,
    model_probs: np.ndarray = None,
) -> float:
    """Compute Total Variation Distance (TVD)."""
    p_data, p_model = compute_distributions(
        ground_truth,
        model_samples,
        n_bins,
        smoothing=0.0,
        exact_probs=exact_probs,
        model_probs=model_probs,
    )
    return 0.5 * np.sum(np.abs(p_data - p_model))


def compute_precision_recall_f1(
    ground_truth: np.ndarray,
    model_samples: np.ndarray,
    sigma: float,
    threshold: float = None,
) -> dict:
    """Compute precision, recall, F1 using kernel-based matching."""
    if threshold is None:
        threshold = np.exp(-1.0)

    K_data_model = fast_binary_gaussian_kernel(ground_truth, model_samples, sigma)
    max_kernel_per_data = np.max(K_data_model, axis=1)
    max_kernel_per_model = np.max(K_data_model, axis=0)

    data_matched_mask = max_kernel_per_data > threshold
    model_matched_mask = max_kernel_per_model > threshold

    n_model_matched = np.sum(model_matched_mask)
    n_data_matched = np.sum(data_matched_mask)

    precision = n_model_matched / len(model_samples) if len(model_samples) > 0 else 0.0
    recall = n_data_matched / len(ground_truth) if len(ground_truth) > 0 else 0.0
    support_match = (np.mean(max_kernel_per_data) + np.mean(max_kernel_per_model)) / 2
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'support_match': float(support_match),
        'f_score': float(f_score),
    }


def compute_metrics(
    ground_truth: np.ndarray,
    model_samples: np.ndarray,
    validity_fn: callable,
    coverage_fn: callable,
) -> dict:
    """Compute validity and coverage metrics."""
    ground_truth = np.asarray(ground_truth)
    model_samples = np.asarray(model_samples)
    validity_rate = validity_fn(model_samples)
    coverage = coverage_fn(ground_truth, model_samples)
    return {'validity_rate': validity_rate, 'coverage': coverage}


def compute_mmd(ground_truth: np.ndarray, samples: np.ndarray, sigma: float | list) -> float:
    """Compute UNBIASED MMD^2 for Gaussian kernel, averaged over sigmas."""
    sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma

    H_gt_gt = compute_hamming_matrix(ground_truth, ground_truth)
    H_s_s = compute_hamming_matrix(samples, samples)
    H_gt_s = compute_hamming_matrix(ground_truth, samples)
    m = len(ground_truth)
    n = len(samples)

    def unbiased_self(H, n_points, k_fn):
        if n_points <= 1:
            return 0.0
        k_mat = k_fn(H)
        return (np.sum(k_mat) - n_points) / (n_points * (n_points - 1))

    mmd_components = []
    for s in sigmas:
        k_fn = lambda h, s=s: np.exp(-h / (2 * s**2))
        mmd_sq = (
            unbiased_self(H_gt_gt, m, k_fn)
            + unbiased_self(H_s_s, n, k_fn)
            - 2 * np.mean(k_fn(H_gt_s))
        )
        mmd_components.append(float(mmd_sq))

    return sum(mmd_components) / len(mmd_components)
