"""Utility functions for IQP boosting with Gaussian kernels."""
import numpy as np
from scipy.special import comb
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


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


def compute_distributions(ground_truth: np.ndarray, model_samples: np.ndarray,
                         n_bins: int = None, smoothing: float = 1e-10, max_qubits: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """Compute probability distributions from discrete samples.

    For <=max_qubits: uses full 2^n histogram.
    For >max_qubits: uses empirical support (only observed bitstrings are binned).
    """
    ground_truth = np.asarray(ground_truth, dtype=int)
    model_samples = np.asarray(model_samples, dtype=int)

    n_features = ground_truth.shape[1]

    if n_features > max_qubits and n_bins is None:
        # Empirical support approach: hash bitstrings to avoid 2^n allocation
        def rows_to_tuples(arr):
            return [tuple(row) for row in arr]

        gt_tuples = rows_to_tuples(ground_truth)
        model_tuples = rows_to_tuples(model_samples)

        # Union of observed bitstrings
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

    # Standard full-histogram approach for <=max_qubits qubits
    def binary_to_int(samples):
        return np.sum(samples * (2 ** np.arange(samples.shape[1])), axis=1)

    gt_ints = binary_to_int(ground_truth)
    model_ints = binary_to_int(model_samples)

    if n_bins is None:
        n_bins = 2 ** n_features

    gt_counts = np.bincount(gt_ints, minlength=n_bins) + smoothing
    model_counts = np.bincount(model_ints, minlength=n_bins) + smoothing

    p_data = gt_counts / gt_counts.sum()
    p_model = model_counts / model_counts.sum()
    return p_data, p_model


def compute_kl_divergence(ground_truth: np.ndarray, model_samples: np.ndarray,
                          n_bins: int = None, smoothing: float = 1e-10) -> float:
    """Compute KL divergence between discrete sample distributions using Scipy."""
    p_data, p_model = compute_distributions(ground_truth, model_samples, n_bins, smoothing)
    return float(entropy(p_data, p_model))


def compute_jsd(ground_truth: np.ndarray, model_samples: np.ndarray,
                n_bins: int = None, smoothing: float = 1e-10) -> float:
    """Compute Jensen-Shannon Distance (metric) using Scipy."""
    p_data, p_model = compute_distributions(ground_truth, model_samples, n_bins, smoothing)
    return float(jensenshannon(p_data, p_model))


def compute_tvd(ground_truth: np.ndarray, model_samples: np.ndarray,
                n_bins: int = None) -> float:
    """Compute Total Variation Distance (TVD)."""
    p_data, p_model = compute_distributions(ground_truth, model_samples, n_bins, smoothing=0.0)
    return 0.5 * np.sum(np.abs(p_data - p_model))


def compute_precision_recall_f1(ground_truth: np.ndarray, model_samples: np.ndarray,
                                sigma: float, threshold: float = None) -> dict:
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


def compute_metrics(ground_truth: np.ndarray, model_samples: np.ndarray,
                    validity_fn: callable, coverage_fn: callable) -> dict:
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
        mmd_sq = (unbiased_self(H_gt_gt, m, k_fn) + 
                  unbiased_self(H_s_s, n, k_fn) - 
                  2 * np.mean(k_fn(H_gt_s)))
        mmd_components.append(float(mmd_sq))

    return sum(mmd_components) / len(mmd_components)


def compute_optimal_alpha_samples(
    samples_old: np.ndarray,
    samples_new: np.ndarray,
    ground_truth: np.ndarray,
    sigma: float | list,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0
) -> float:
    """Analytically compute the optimal mixing coefficient alpha using SAMPLES."""
    sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
    
    num = 0.0
    den = 0.0

    H_oo = compute_hamming_matrix(samples_old, samples_old)
    H_nn = compute_hamming_matrix(samples_new, samples_new)
    H_on = compute_hamming_matrix(samples_old, samples_new)
    H_od = compute_hamming_matrix(samples_old, ground_truth)
    H_nd = compute_hamming_matrix(samples_new, ground_truth)

    n_o = len(samples_old)
    n_n = len(samples_new)

    for s in sigmas:
        k_fn = lambda h, s=s: np.exp(-h / (2 * s**2))

        s_old = 0.0 if n_o <= 1 else (np.sum(k_fn(H_oo)) - n_o) / (n_o * (n_o - 1))
        s_new = 0.0 if n_n <= 1 else (np.sum(k_fn(H_nn)) - n_n) / (n_n * (n_n - 1))

        c_old_new = np.mean(k_fn(H_on))
        c_old_data = np.mean(k_fn(H_od))
        c_new_data = np.mean(k_fn(H_nd))

        den += (s_old + s_new - 2 * c_old_new)
        num += (s_old - c_old_new + c_new_data - c_old_data)

    if den <= 1e-12:
        return alpha_max if (den - 2 * num) < 0 else alpha_min

    alpha_opt = num / den
    return float(np.clip(alpha_opt, alpha_min, alpha_max))


def compute_optimal_weights_qp(
    all_trs: list,
    trs_data: list[np.ndarray],
    all_corrs: list = None,
    n_samples: int = 1000
) -> np.ndarray:
    """Fully corrective weight optimization via QP."""
    from scipy.optimize import minimize

    M = len(all_trs)
    if M == 0:
        return np.array([])
    if M == 1:
        return np.array([1.0])

    n_sigmas = len(trs_data)

    K_mm = np.zeros((M, M))
    k_md = np.zeros(M)

    for s_idx in range(n_sigmas):
        t_data = np.asarray(trs_data[s_idx])
        t_models = [np.asarray(all_trs[m][s_idx]) for m in range(M)]

        for i in range(M):
            k_md[i] += np.mean(t_models[i] * t_data)

            for j in range(i, M):
                if i == j:
                    if all_corrs is not None:
                        cov_i = np.mean(np.asarray(all_corrs[i][s_idx]))
                        s_ii = (np.mean(t_models[i]**2) - cov_i) * n_samples / (n_samples - 1)
                    else:
                        s_ii = np.mean(t_models[i]**2)
                    K_mm[i, i] += s_ii
                else:
                    c_ij = np.mean(t_models[i] * t_models[j])
                    K_mm[i, j] += c_ij
                    K_mm[j, i] += c_ij

    # Average over sigmas
    K_mm /= n_sigmas
    k_md /= n_sigmas

    def objective(w):
        return w @ K_mm @ w - 2.0 * w @ k_md

    def gradient(w):
        return 2.0 * K_mm @ w - 2.0 * k_md

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(M)}
    bounds = [(0.0, 1.0)] * M
    w0 = np.ones(M) / M

    result = minimize(
        objective, w0, jac=gradient, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 500}
    )

    if not result.success:
        print(f"  [Warning] QP optimization did not converge: {result.message}")

    w_opt = np.maximum(result.x, 0.0)
    w_opt /= w_opt.sum()
    return w_opt


def compute_optimal_alpha_dual(
    trs_old: list[np.ndarray],
    trs_new: list[np.ndarray],
    trs_data: list[np.ndarray],
    trs_corr_old: list[np.ndarray] = None,
    trs_corr_new: list[np.ndarray] = None,
    n_samples: int = 1000,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0
) -> float:
    """Analytically compute the optimal mixing coefficient alpha in the Dual Space."""
    num = 0.0
    den = 0.0
    n_sigmas = len(trs_data)

    for i in range(n_sigmas):
        t_old = np.asarray(trs_old[i])
        t_new = np.asarray(trs_new[i])
        t_data = np.asarray(trs_data[i])

        if trs_corr_old is not None:
            cov_old = np.mean(trs_corr_old[i])
            s_old = (np.mean(t_old**2) - cov_old) * n_samples / (n_samples - 1)
        else:
            s_old = np.mean(t_old**2)

        if trs_corr_new is not None:
            cov_new = np.mean(trs_corr_new[i])
            s_new = (np.mean(t_new**2) - cov_new) * n_samples / (n_samples - 1)
        else:
            s_new = np.mean(t_new**2)

        c_old_new = np.mean(t_old * t_new)
        c_old_data = np.mean(t_old * t_data)
        c_new_data = np.mean(t_new * t_data)

        den += (s_old + s_new - 2 * c_old_new)
        num += (s_old - c_old_new + c_new_data - c_old_data)

    if den <= 1e-12:
        return alpha_max if (den - 2 * num) < 0 else alpha_min

    alpha_opt = num / den
    return float(np.clip(alpha_opt, alpha_min, alpha_max))