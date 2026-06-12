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


def _enumerate_rect_masks(nq: int, H: int, W: int,
                           max_pw: int, max_ph: int) -> list[np.ndarray]:
    """Return list of bool masks (nq,) for all valid centered odd rectangles."""
    rows = np.arange(nq) // W
    cols = np.arange(nq) % W
    rects = []
    for w in range(1, max_pw + 1, 2):
        for h in range(1, max_ph + 1, 2):
            half_w = (w - 1) // 2
            half_h = (h - 1) // 2
            for cx in range(half_w, W - half_w):
                for cy in range(half_h, H - half_h):
                    r0 = cy - half_h
                    r1 = cy + half_h + 1
                    c0 = cx - half_w
                    c1 = cx + half_w + 1
                    mask = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
                    rects.append(mask)
    return rects


def _kernel_conv_parity(
    gt: np.ndarray, samples: np.ndarray,
    rects: list[np.ndarray],
) -> np.ndarray:
    """K_conv(x,y) = E_rect[(-1)^popcount(x⊕y within rect)].

    Uses the parity trick: returns an (m, n) kernel matrix.
    """
    m, n = len(gt), len(samples)
    n_rects = len(rects)
    if n_rects == 0:
        return np.ones((m, n))

    gt_bits = np.asarray(gt, dtype=np.uint8)
    sample_bits = np.asarray(samples, dtype=np.uint8)

    def parities(bits):
        P = np.empty((bits.shape[0], n_rects), dtype=bool)
        for r_idx, mask in enumerate(rects):
            pop = bits[:, mask].sum(axis=1)
            P[:, r_idx] = (pop % 2).astype(bool)
        return P

    P_gt = parities(gt_bits)
    P_s = parities(samples)
    H_rect = (P_gt[:, None, :] ^ P_s[None, :, :]).sum(axis=-1)
    return 1.0 - 2.0 * H_rect.astype(np.float64) / n_rects


def _kernel_conv_gauss(
    gt: np.ndarray, samples: np.ndarray,
    rects: list[np.ndarray],
    sigma_list: list[float],
) -> np.ndarray:
    """K_conv(x,y) = E_rect[exp(-d_H(x_P,y_P) / 2σ²)], averaged over sigmas.

    Returns an (m, n) kernel matrix.
    """
    m, n = len(gt), len(samples)
    n_rects = len(rects)
    if n_rects == 0:
        return np.ones((m, n))

    gt_bits = np.asarray(gt, dtype=np.uint8)
    sample_bits = np.asarray(samples, dtype=np.uint8)

    K = np.zeros((m, n), dtype=np.float64)
    for mask in rects:
        gt_p = gt_bits[:, mask]
        smp_p = sample_bits[:, mask]
        H = (gt_p[:, None, :] ^ smp_p[None, :, :]).sum(axis=-1)
        for sigma in sigma_list:
            K += np.exp(-H.astype(np.float64) / (2.0 * sigma**2))
    K /= n_rects * len(sigma_list)
    return K


def compute_mmd(ground_truth: np.ndarray, samples: np.ndarray,
                sigma: float | list | dict) -> float:
    """Compute UNBIASED MMD^2, averaged over sigmas.

    For a dict sigma of type ``"spatial_gaussian_mixture"``, computes
    ``λ·MMD²_conv + (1-λ)·MMD²_gauss`` where:

    - If ``kernel="parity"`` (default): ``K_conv = E_rect[(-1)^{popcount}]``
    - If ``kernel="gaussian"``: ``K_conv = E_rect[exp(-d_H / 2σ²)]``
    """
    if isinstance(sigma, dict):
        stype = sigma.get('type', '')
        if stype == 'mkl':
            from src.core.mkl import (
                enumerate_all_base_kernels, compute_operator_expectations,
                compute_mmd2_from_expectations, _FORMULA_KERNEL_NAMES,
            )
            alphas = sigma['mkl_weights']
            base_kernels = sigma['mkl_base_kernels']
            grid_shape = tuple(sigma['grid_shape'])
            nq = ground_truth.shape[1]
            geo_kernels = [k for k in base_kernels if k not in _FORMULA_KERNEL_NAMES]
            kernel_masks = enumerate_all_base_kernels(grid_shape[0], grid_shape[1], names=geo_kernels)
            total = 0.0
            for name, alpha in zip(base_kernels, alphas):
                if alpha <= 0:
                    continue
                if name in _FORMULA_KERNEL_NAMES:
                    if name == "gaussian_mixture":
                        _sigmas = sigma.get('sigma', None)
                        sl = _sigmas if isinstance(_sigmas, list) else ([_sigmas] if isinstance(_sigmas, (int, float)) else None)
                        if sl is not None:
                            total += alpha * compute_mmd(ground_truth, samples, sl)
                    elif name == "spatial_rect":
                        lam = sigma.get('lambda', 0.0)
                        ker = sigma.get('kernel', 'parity')
                        mpw = sigma.get('max_patch_width', 3)
                        mph = sigma.get('max_patch_height', 3)
                        rects = _enumerate_rect_masks(nq, *grid_shape, mpw, mph)
                        if ker == "parity":
                            K_pp = _kernel_conv_parity(ground_truth, ground_truth, rects)
                            K_ss = _kernel_conv_parity(samples, samples, rects)
                            K_ps = _kernel_conv_parity(ground_truth, samples, rects)
                        else:
                            _sl = sigma.get('sigma', [1.0])
                            sl2 = _sl if isinstance(_sl, list) else [_sl]
                            K_pp = _kernel_conv_gauss(ground_truth, ground_truth, rects, sl2)
                            K_ss = _kernel_conv_gauss(samples, samples, rects, sl2)
                            K_ps = _kernel_conv_gauss(ground_truth, samples, rects, sl2)
                        m, n = len(ground_truth), len(samples)
                        usp = (np.sum(K_pp) - m) / (m * (m - 1)) if m > 1 else 0.0
                        uss = (np.sum(K_ss) - n) / (n * (n - 1)) if n > 1 else 0.0
                        total += alpha * (usp + uss - 2 * np.mean(K_ps))
                else:
                    ops = kernel_masks.get(name)
                    if ops is None or ops.shape[0] == 0:
                        continue
                    E_P = compute_operator_expectations(ground_truth, ops)
                    E_Q = compute_operator_expectations(samples, ops)
                    total += alpha * compute_mmd2_from_expectations(E_P, E_Q)
            return total

        lam = sigma.get('lambda', 0.0)
        kernel = sigma.get('kernel', 'parity')
        grid_shape = tuple(sigma['grid_shape'])
        max_pw = sigma.get('max_patch_width', 3)
        max_ph = sigma.get('max_patch_height', 3)
        sigma_list = sigma['sigma']
        if isinstance(sigma_list, (int, float)):
            sigma_list = [sigma_list]

        mmd_gauss = compute_mmd(ground_truth, samples, sigma_list)

        nq = ground_truth.shape[1]
        rects = _enumerate_rect_masks(nq, *grid_shape, max_pw, max_ph)
        if kernel == "parity":
            K_gt_gt = _kernel_conv_parity(ground_truth, ground_truth, rects)
            K_s_s = _kernel_conv_parity(samples, samples, rects)
            K_gt_s = _kernel_conv_parity(ground_truth, samples, rects)
        elif kernel == "gaussian":
            K_gt_gt = _kernel_conv_gauss(ground_truth, ground_truth, rects, sigma_list)
            K_s_s = _kernel_conv_gauss(samples, samples, rects, sigma_list)
            K_gt_s = _kernel_conv_gauss(ground_truth, samples, rects, sigma_list)
        else:
            raise ValueError(f"Unknown spatial kernel: {kernel}")
        m, n = len(ground_truth), len(samples)
        unbiased_self_gt = (np.sum(K_gt_gt) - m) / (m * (m - 1)) if m > 1 else 0.0
        unbiased_self_s = (np.sum(K_s_s) - n) / (n * (n - 1)) if n > 1 else 0.0
        mmd_conv = unbiased_self_gt + unbiased_self_s - 2 * np.mean(K_gt_s)

        return lam * float(mmd_conv) + (1.0 - lam) * float(mmd_gauss)

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
