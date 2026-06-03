"""Mixture weighting optimization helpers for boosted ensembles."""

import numpy as np

from src.core.metrics import compute_hamming_matrix


def compute_optimal_alpha_samples(
    samples_old: np.ndarray,
    samples_new: np.ndarray,
    ground_truth: np.ndarray,
    sigma: float | list,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> float:
    """Analytically compute the optimal mixing coefficient alpha using samples."""
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

        den += s_old + s_new - 2 * c_old_new
        num += s_old - c_old_new + c_new_data - c_old_data

    if den <= 1e-12:
        return alpha_max if (den - 2 * num) < 0 else alpha_min

    alpha_opt = num / den
    return float(np.clip(alpha_opt, alpha_min, alpha_max))


def compute_optimal_alpha_tvd_samples(
    samples_old: np.ndarray,
    samples_new: np.ndarray,
    ground_truth: np.ndarray,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> float:
    """Find optimal mixing alpha minimising empirical TVD wrt ground truth."""
    from scipy.optimize import minimize_scalar

    samples_old = np.asarray(samples_old, dtype=int)
    samples_new = np.asarray(samples_new, dtype=int)
    ground_truth = np.asarray(ground_truth, dtype=int)
    n_features = ground_truth.shape[1]

    if n_features <= 18:
        powers = 2 ** np.arange(n_features)

        def to_int(arr):
            return (arr * powers).sum(axis=1)

        n_bins = 2 ** n_features
        p_data = np.bincount(to_int(ground_truth), minlength=n_bins).astype(float)
        p_old = np.bincount(to_int(samples_old), minlength=n_bins).astype(float)
        p_new = np.bincount(to_int(samples_new), minlength=n_bins).astype(float)
    else:
        def rows_to_tuples(arr):
            return [tuple(row) for row in arr]

        all_keys = sorted(
            set(rows_to_tuples(ground_truth))
            | set(rows_to_tuples(samples_old))
            | set(rows_to_tuples(samples_new))
        )
        key_to_idx = {k: i for i, k in enumerate(all_keys)}
        n_bins = len(all_keys)

        def count_array(arr):
            c = np.zeros(n_bins)
            for t in rows_to_tuples(arr):
                c[key_to_idx[t]] += 1
            return c

        p_data = count_array(ground_truth)
        p_old = count_array(samples_old)
        p_new = count_array(samples_new)

    p_data /= p_data.sum()
    p_old /= p_old.sum()
    p_new /= p_new.sum()

    delta = p_new - p_old

    def tvd_objective(alpha):
        return 0.5 * np.sum(np.abs(p_data - p_old - alpha * delta))

    result = minimize_scalar(tvd_objective, bounds=(alpha_min, alpha_max), method='bounded')
    return float(np.clip(result.x, alpha_min, alpha_max))


def _mix_samples_by_slice(
    samples_old: np.ndarray,
    samples_new: np.ndarray,
    alpha: float,
    n_total: int,
) -> np.ndarray:
    """Return a mixed sample array of length n_total from pre-drawn old/new samples."""
    samples_old = np.asarray(samples_old, dtype=np.int8)
    samples_new = np.asarray(samples_new, dtype=np.int8)

    n_new = int(round(alpha * n_total))
    n_old = n_total - n_new
    old_slice = samples_old[:min(n_old, len(samples_old))]
    new_slice = samples_new[:min(n_new, len(samples_new))]
    parts = [p for p in (old_slice, new_slice) if len(p) > 0]
    mixed = np.vstack(parts) if len(parts) > 1 else parts[0]
    return np.asarray(mixed, dtype=np.int8)


def compute_optimal_alpha_validity(
    samples_old: np.ndarray,
    samples_new: np.ndarray,
    validity_fn: callable,
    n_grid: int = 11,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> float:
    """Find alpha maximising the validity rate of the mixed ensemble."""
    samples_old = np.asarray(samples_old, dtype=np.int8)
    samples_new = np.asarray(samples_new, dtype=np.int8)
    n_total = min(len(samples_old), len(samples_new))
    alphas = np.linspace(alpha_min, alpha_max, n_grid)

    best_alpha, best_val = alpha_min, -np.inf
    for alpha in alphas:
        mix = _mix_samples_by_slice(samples_old, samples_new, alpha, n_total)
        val = validity_fn(mix)
        if val > best_val:
            best_val = val
            best_alpha = alpha

    return float(best_alpha)


def compute_optimal_alpha_coverage(
    samples_old: np.ndarray,
    samples_new: np.ndarray,
    ground_truth: np.ndarray,
    coverage_fn: callable,
    n_grid: int = 11,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> float:
    """Find alpha maximising coverage of the mixed ensemble."""
    samples_old = np.asarray(samples_old, dtype=np.int8)
    samples_new = np.asarray(samples_new, dtype=np.int8)
    ground_truth = np.asarray(ground_truth, dtype=np.int8)
    n_total = min(len(samples_old), len(samples_new))
    alphas = np.linspace(alpha_min, alpha_max, n_grid)

    best_alpha, best_cov = alpha_min, -np.inf
    for alpha in alphas:
        mix = _mix_samples_by_slice(samples_old, samples_new, alpha, n_total)
        cov = coverage_fn(ground_truth, mix)
        if cov > best_cov:
            best_cov = cov
            best_alpha = alpha

    return float(best_alpha)


def compute_optimal_alpha_validity_coverage_sum(
    samples_old: np.ndarray,
    samples_new: np.ndarray,
    ground_truth: np.ndarray,
    validity_fn: callable,
    coverage_fn: callable,
    validity_weight: float = 0.5,
    n_grid: int = 11,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> float:
    """Find alpha maximizing weighted sum of validity and coverage."""
    samples_old = np.asarray(samples_old, dtype=np.int8)
    samples_new = np.asarray(samples_new, dtype=np.int8)
    ground_truth = np.asarray(ground_truth, dtype=np.int8)

    w = float(np.clip(validity_weight, 0.0, 1.0))
    n_grid = max(2, int(n_grid))
    n_total = min(len(samples_old), len(samples_new))
    alphas = np.linspace(alpha_min, alpha_max, n_grid)

    best_alpha, best_obj = alpha_min, -np.inf
    for alpha in alphas:
        mix = _mix_samples_by_slice(samples_old, samples_new, alpha, n_total)
        val = validity_fn(mix)
        cov = coverage_fn(ground_truth, mix)
        obj = w * val + (1.0 - w) * cov
        if obj > best_obj:
            best_obj = obj
            best_alpha = alpha

    return float(best_alpha)


def compute_optimal_weights_qp(
    all_trs: list,
    trs_data: list[np.ndarray],
    all_corrs: list = None,
    n_samples: int = 1000,
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
        objective,
        w0,
        jac=gradient,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 500},
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
    alpha_max: float = 1.0,
) -> float:
    """Analytically compute the optimal mixing coefficient alpha in dual space."""
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

        den += s_old + s_new - 2 * c_old_new
        num += s_old - c_old_new + c_new_data - c_old_data

    if den <= 1e-12:
        return alpha_max if (den - 2 * num) < 0 else alpha_min

    alpha_opt = num / den
    return float(np.clip(alpha_opt, alpha_min, alpha_max))
