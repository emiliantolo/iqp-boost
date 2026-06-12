"""Multiple Kernel Learning (MKL) for spatial-aware MMD.

Provides:
1. Geometric operator enumeration for the kernel dictionary
2. JIT-compatible MKL operator sampling (build_mkl_ops)
3. Standalone MKL optimization (maximise MMD test power)

Each base kernel is defined by an operator set S_m — a specific subset of
Pauli-Z strings on a 2D grid. The total kernel is K_α = Σ α_m · K_m and
MMD²_α = Σ α_m · MMD²_m.

The α* weights are found by maximising the Sutherland test statistic:
  α* = argmax_{α≥0}  αᵀv / √(αᵀΣα + λ‖α‖²)
where v_m = MMD²_m(P_data, Q_fake) and Σ is the covariance of MMD² estimates.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from src.core.importance_sampling import make_ops_from_pmf, _resolve_pmfs


# ---------------------------------------------------------------------------
# Base kernel names (used as keys in config and dicts)
# ---------------------------------------------------------------------------

BASE_KERNEL_NAMES = [
    "density",          # 1-body: Z_i for every pixel
    "nn_h",             # 2-body: Z_i Z_j for horizontal nearest neighbours
    "nn_v",             # 2-body: Z_i Z_j for vertical nearest neighbours
    "nnn",              # 2-body: Z_i Z_j for next-nearest neighbours (diagonal)
    "plaquette",        # 4-body: Z_i Z_j Z_k Z_l per 2×2 block
    "global_parity",    # N-body: ⨂Z_i over all pixels
    "gaussian_mixture", # Multi-sigma Gaussian RBF kernel (formula-based, not enumerated)
    "spatial_rect",     # Spatial rect patches with parity kernel (enumerated rect masks)
]


# ---------------------------------------------------------------------------
# Geometric operator enumeration
# ---------------------------------------------------------------------------

def enumerate_density_ops(H: int, W: int) -> np.ndarray:
    """Enumerate single-qubit Z_i operators: (H*W, H*W) one-hot identity."""
    n = H * W
    return np.eye(n, dtype=np.uint8)


def enumerate_nn_horizontal_ops(H: int, W: int) -> np.ndarray:
    """Enumerate horizontal nearest-neighbour Z_i Z_j operators."""
    n = H * W
    ops = []
    for i in range(H):
        for j in range(W - 1):
            mask = np.zeros(n, dtype=np.uint8)
            mask[i * W + j] = 1
            mask[i * W + (j + 1)] = 1
            ops.append(mask)
    return np.stack(ops) if ops else np.zeros((0, n), dtype=np.uint8)


def enumerate_nn_vertical_ops(H: int, W: int) -> np.ndarray:
    """Enumerate vertical nearest-neighbour Z_i Z_j operators."""
    n = H * W
    ops = []
    for i in range(H - 1):
        for j in range(W):
            mask = np.zeros(n, dtype=np.uint8)
            mask[i * W + j] = 1
            mask[(i + 1) * W + j] = 1
            ops.append(mask)
    return np.stack(ops) if ops else np.zeros((0, n), dtype=np.uint8)


def enumerate_nnn_ops(H: int, W: int) -> np.ndarray:
    """Enumerate next-nearest-neighbour (diagonal) Z_i Z_j operators."""
    n = H * W
    ops = []
    for i in range(H - 1):
        for j in range(W - 1):
            # bottom-right diagonal
            mask1 = np.zeros(n, dtype=np.uint8)
            mask1[i * W + j] = 1
            mask1[(i + 1) * W + (j + 1)] = 1
            ops.append(mask1)
            # bottom-left diagonal
            if j > 0:
                mask2 = np.zeros(n, dtype=np.uint8)
                mask2[i * W + j] = 1
                mask2[(i + 1) * W + (j - 1)] = 1
                ops.append(mask2)
    return np.stack(ops) if ops else np.zeros((0, n), dtype=np.uint8)


def enumerate_plaquette_ops(H: int, W: int) -> np.ndarray:
    """Enumerate 2×2 plaquette operators Z_i Z_j Z_k Z_l."""
    n = H * W
    ops = []
    for i in range(H - 1):
        for j in range(W - 1):
            mask = np.zeros(n, dtype=np.uint8)
            mask[i * W + j] = 1
            mask[i * W + (j + 1)] = 1
            mask[(i + 1) * W + j] = 1
            mask[(i + 1) * W + (j + 1)] = 1
            ops.append(mask)
    return np.stack(ops) if ops else np.zeros((0, n), dtype=np.uint8)


def enumerate_global_parity_ops(H: int, W: int) -> np.ndarray:
    """Enumerate global parity operator ⨂Z_i: single all-ones mask."""
    n = H * W
    return np.ones((1, n), dtype=np.uint8)


# Geometric kernels that can be fully enumerated as Pauli-Z masks.
# ``gaussian_mixture`` and ``spatial_rect`` are handled separately
# (formula-based MMD², sampling-based operator generation).
_GEOMETRIC_KERNEL_NAMES = ["density", "nn_h", "nn_v", "nnn", "plaquette", "global_parity"]

# For geometric kernels: fully enumerated as Pauli-Z masks.
_ENUMERATORS = {
    "density": enumerate_density_ops,
    "nn_h": enumerate_nn_horizontal_ops,
    "nn_v": enumerate_nn_vertical_ops,
    "nnn": enumerate_nnn_ops,
    "plaquette": enumerate_plaquette_ops,
    "global_parity": enumerate_global_parity_ops,
}

# Kernels whose MMD² is computed via a direct kernel formula (not operator expectations).
_FORMULA_KERNEL_NAMES = ["gaussian_mixture", "spatial_rect"]


def enumerate_base_kernel_ops(kernel_name: str, H: int, W: int) -> np.ndarray:
    """Return the binary mask matrix (n_ops, n_qubits) for a base kernel.

    Only geometric kernels are enumerated. Formula-based kernels
    (gaussian_mixture, spatial_rect) raise ValueError.
    """
    fn = _ENUMERATORS.get(kernel_name)
    if fn is None:
        raise ValueError(
            f"Cannot enumerate '{kernel_name}'. "
            f"Geometric kernels: {list(_ENUMERATORS)}. "
            f"Formula-based kernels: {_FORMULA_KERNEL_NAMES} (use compute_*_stats instead)."
        )
    return fn(H, W)


def enumerate_all_base_kernels(H: int, W: int, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Enumerate operator sets for all *geometric* base kernels.

    Formula-based kernels (gaussian_mixture, spatial_rect) are excluded.
    Returns dict mapping kernel name → (n_ops, n_qubits) binary matrix.
    """
    if names is None:
        names = BASE_KERNEL_NAMES
    result = {}
    for name in names:
        if name in _FORMULA_KERNEL_NAMES:
            continue
        result[name] = enumerate_base_kernel_ops(name, H, W)
    return result


# ---------------------------------------------------------------------------
# Spatial rect enumeration (for spatial_rect base kernel)
# ---------------------------------------------------------------------------

def enumerate_spatial_rect_masks(H: int, W: int, max_pw: int = 4, max_ph: int = 4) -> np.ndarray:
    """Enumerate all rect patches as binary masks on a 2D grid.

    Returns (n_rects, H*W) uint8 masks where 1 = inside the rect.
    """
    n = H * W
    rects = []
    max_pw = min(max_pw, W)
    max_ph = min(max_ph, H)
    for pw in range(1, max_pw + 1):
        for ph in range(1, max_ph + 1):
            for cy in range(H - ph + 1):
                for cx in range(W - pw + 1):
                    mask = np.zeros(n, dtype=np.uint8)
                    for dy in range(ph):
                        for dx in range(pw):
                            mask[(cy + dy) * W + (cx + dx)] = 1
                    rects.append(mask)
    return np.stack(rects) if rects else np.zeros((0, n), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Expectation computation
# ---------------------------------------------------------------------------

def compute_operator_expectations(data: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """Compute E_data[Z_s] for each operator s.

    Z_s(x) = (-1)^{s·x mod 2}. Returns vector of length n_ops.
    """
    return np.mean(1 - 2 * ((data @ ops.T) % 2), axis=0)


def compute_mmd2_from_expectations(E_P: np.ndarray, E_Q: np.ndarray) -> float:
    """MMD² = Σ_s (E_P[Z_s] - E_Q[Z_s])² for a single base kernel."""
    return float(np.sum((E_P - E_Q) ** 2))


def compute_mmd2_variance(E_P: np.ndarray, E_Q: np.ndarray,
                          data: np.ndarray, qfake: np.ndarray) -> float:
    """Estimate Var[MMD²] via Sutherland U-statistic on sample-level contributions."""
    m = len(data)
    n = len(qfake)

    Z_data = 1 - 2 * ((data @ ops.T) % 2)
    Z_q = 1 - 2 * ((qfake @ ops.T) % 2)

    var_data = np.var(Z_data, axis=0, ddof=1)
    var_q = np.var(Z_q, axis=0, ddof=1)

    diff = E_P - E_Q
    var_per_op = 4 * diff ** 2 * (var_data / m + var_q / n)
    return float(np.sum(var_per_op))


# ---------------------------------------------------------------------------
# Formula-based kernel MMD² and variance (gaussian_mixture, spatial_rect)
# ---------------------------------------------------------------------------

def _compute_gaussian_kernel_matrices(
    data: np.ndarray, qfake: np.ndarray,
    sigma_list: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute multi-sigma Gaussian kernel matrices.

    Uses the identity hamming(x,y) = sum(x) + sum(y) - 2*(x·y) to avoid
    O(n² × nq) intermediate 3D arrays.

    Returns (K_pp, K_qq, K_pq) averaged over sigmas.
    """
    m, n = len(data), len(qfake)
    sigmas = sigma_list if isinstance(sigma_list, list) else [sigma_list]
    n_sigmas = len(sigmas)

    data_bits = data.astype(np.float64)
    qfake_bits = qfake.astype(np.float64)

    cnt_p = np.sum(data_bits, axis=1)   # (m,)
    cnt_q = np.sum(qfake_bits, axis=1)  # (n,)

    # h(i,j) = cnt(i) + cnt(j) - 2 * dot(i,j)
    h_pp = cnt_p[:, None] + cnt_p[None, :] - 2.0 * (data_bits @ data_bits.T)
    h_qq = cnt_q[:, None] + cnt_q[None, :] - 2.0 * (qfake_bits @ qfake_bits.T)
    h_pq = cnt_p[:, None] + cnt_q[None, :] - 2.0 * (data_bits @ qfake_bits.T)

    K_pp = np.zeros((m, m), dtype=np.float64)
    K_qq = np.zeros((n, n), dtype=np.float64)
    K_pq = np.zeros((m, n), dtype=np.float64)

    for sigma in sigmas:
        s2 = sigma ** 2
        K_pp += np.exp(-h_pp / (2.0 * s2))
        K_qq += np.exp(-h_qq / (2.0 * s2))
        K_pq += np.exp(-h_pq / (2.0 * s2))

    K_pp /= n_sigmas
    K_qq /= n_sigmas
    K_pq /= n_sigmas
    return K_pp, K_qq, K_pq


def _unbiased_mmd2_from_kernels(
    K_pp: np.ndarray, K_qq: np.ndarray, K_pq: np.ndarray,
) -> float:
    """Unbiased MMD² from pre-computed kernel matrices."""
    m, n = K_pp.shape[0], K_qq.shape[0]
    unbiased_self_p = (np.sum(K_pp) - m) / (m * (m - 1)) if m > 1 else 0.0
    unbiased_self_q = (np.sum(K_qq) - n) / (n * (n - 1)) if n > 1 else 0.0
    return max(unbiased_self_p + unbiased_self_q - 2 * np.mean(K_pq), 0.0)


def compute_gaussian_mmd2_stats(
    data: np.ndarray, qfake: np.ndarray,
    sigma_list: list[float],
) -> tuple[float, float, tuple]:
    """Compute MMD² and its variance for the multi-sigma Gaussian RBF kernel.

    Returns (mmd2, variance, (K_pp, K_qq, K_pq)) for reuse in bootstrap.
    """
    K_pp, K_qq, K_pq = _compute_gaussian_kernel_matrices(data, qfake, sigma_list)
    mmd2 = _unbiased_mmd2_from_kernels(K_pp, K_qq, K_pq)

    m, n = len(data), len(qfake)
    xi = np.mean(K_pp, axis=1) - np.mean(K_pq, axis=1)
    yi = np.mean(K_qq, axis=1) - np.mean(K_pq, axis=0)
    pooled = np.concatenate([xi, yi])
    var_estimate = max(np.var(pooled, ddof=1) / len(pooled), 1e-15)
    return mmd2, var_estimate, (K_pp, K_qq, K_pq)


def _bootstrap_gaussian_mmd2(
    K_pp: np.ndarray, K_qq: np.ndarray, K_pq: np.ndarray,
    idx_p: np.ndarray, idx_q: np.ndarray,
) -> float:
    """Fast bootstrap: re-index pre-computed kernel matrices."""
    K_pp_b = K_pp[np.ix_(idx_p, idx_p)]
    K_qq_b = K_qq[np.ix_(idx_q, idx_q)]
    K_pq_b = K_pq[np.ix_(idx_p, idx_q)]
    return _unbiased_mmd2_from_kernels(K_pp_b, K_qq_b, K_pq_b)


def compute_spatial_rect_stats(
    data: np.ndarray, qfake: np.ndarray,
    grid_shape: tuple[int, int],
    max_pw: int = 4, max_ph: int = 4,
) -> tuple[float, float, np.ndarray]:
    """Compute MMD² and variance for the spatial rect kernel.

    Uses parity kernel: K_conv(x,y) = 1 - 2*H_rect(x,y)/n_rects
    where H_rect is the number of rects where parity differs.

    Returns (mmd2, variance, rect_masks).
    """
    from src.core.metrics import _enumerate_rect_masks, _kernel_conv_parity
    H, W = grid_shape
    nq = data.shape[1]
    rects = _enumerate_rect_masks(nq, H, W, max_pw, max_ph)
    n_rects = len(rects)
    if n_rects == 0:
        return 0.0, 0.0, np.zeros((0, nq), dtype=np.uint8)

    K_pp = _kernel_conv_parity(data, data, rects)
    K_qq = _kernel_conv_parity(qfake, qfake, rects)
    K_pq = _kernel_conv_parity(data, qfake, rects)
    m, n = len(data), len(qfake)

    unbiased_self_p = (np.sum(K_pp) - m) / (m * (m - 1)) if m > 1 else 0.0
    unbiased_self_q = (np.sum(K_qq) - n) / (n * (n - 1)) if n > 1 else 0.0
    mmd2 = unbiased_self_p + unbiased_self_q - 2 * np.mean(K_pq)

    # Variance via per-sample decomposition
    xi = np.mean(K_pp, axis=1) - np.mean(K_pq, axis=1)
    yi = np.mean(K_qq, axis=1) - np.mean(K_pq, axis=0)
    pooled = np.concatenate([xi, yi])
    var_estimate = max(np.var(pooled, ddof=1) / len(pooled), 1e-15)

    # Enumerate rect masks for operator-based covariance computations
    rect_masks = enumerate_spatial_rect_masks(H, W, max_pw, max_ph)
    return mmd2, var_estimate, rect_masks


# ---------------------------------------------------------------------------
# MKL optimisation
# ---------------------------------------------------------------------------

def generate_q_fake(data: np.ndarray, method: str = "scramble",
                    noise_rate: float = 0.2, seed: int = 42) -> np.ndarray:
    """Generate a corrupted baseline distribution Q_fake from data.

    Methods:
      - "scramble": random permutation of pixels within each sample
      - "thermal": flip each pixel with probability noise_rate
      - "dropout": zero out random patches
    """
    rng = np.random.default_rng(seed)
    n, d = data.shape

    if method == "scramble":
        qfake = np.array([rng.permutation(row) for row in data])
    elif method == "thermal":
        noise = rng.binomial(1, noise_rate, size=data.shape)
        qfake = np.abs(data.astype(np.int8) - noise.astype(np.int8))
    elif method == "dropout":
        qfake = data.copy()
        H = W = int(np.sqrt(d))
        patch_size = max(1, min(H, W) // 4)
        for i in range(n):
            if rng.uniform() < 0.5:
                cx = rng.integers(0, W - patch_size + 1)
                cy = rng.integers(0, H - patch_size + 1)
                img = qfake[i].reshape(H, W)
                img[cy:cy + patch_size, cx:cx + patch_size] = 0
                qfake[i] = img.ravel()
    else:
        raise ValueError(f"Unknown Q_fake method: {method}")

    return qfake.astype(np.int8)


def compute_mkl_statistics(
    data: np.ndarray,
    qfake: np.ndarray,
    kernel_masks: dict[str, np.ndarray],
    all_base_kernels: list[str] | None = None,
    sigma_list: list[float] | None = None,
    grid_shape: tuple[int, int] | None = None,
    max_pw: int = 4,
    max_ph: int = 4,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute v vector and Σ covariance matrix for MKL.

    Supports both geometric kernels (from kernel_masks) and formula-based
    kernels (gaussian_mixture, spatial_rect).

    Args:
        data: Training data (m, n_qubits).
        qfake: Corrupted baseline (n, n_qubits).
        kernel_masks: Dict of geometric kernel_name → (n_ops, n_qubits) masks.
        all_base_kernels: Full list of requested base kernel names (incl.
                          formula-based). Used to determine formula_names.
        sigma_list: Sigma values for gaussian_mixture and spatial_rect.
        grid_shape: (H, W) grid dimensions.
        max_pw, max_ph: Max rect patch size for spatial_rect.
        n_bootstrap: Bootstrap iterations for covariance estimation.
        seed: Random seed.

    Returns:
        v: (M,) vector of MMD² per base kernel.
        Sigma: (M, M) covariance matrix.
    """
    m, nq = data.shape
    n_qf = len(qfake)
    rng = np.random.default_rng(seed)

    # ---- collect all kernel names in order ----
    # ``geo_names`` are those that were enumerated (have masks).
    # ``formula_names`` are the formula-based kernels requested by the caller.
    geo_names = list(kernel_masks.keys())
    if all_base_kernels is None:
        all_base_kernels = geo_names + _FORMULA_KERNEL_NAMES
    formula_names = [k for k in _FORMULA_KERNEL_NAMES if k in all_base_kernels]
    names = geo_names + [k for k in formula_names if k not in geo_names]
    M = len(names)
    name_to_idx = {n: i for i, n in enumerate(names)}

    # ---- compute v on full data ----
    v = np.zeros(M, dtype=np.float64)
    for name in geo_names:
        ops = kernel_masks.get(name)
        if ops is not None and ops.shape[0] > 0:
            E_P = compute_operator_expectations(data, ops)
            E_Q = compute_operator_expectations(qfake, ops)
            v[name_to_idx[name]] = compute_mmd2_from_expectations(E_P, E_Q)

    # Pre-compute kernel matrices for fast bootstrap
    gauss_matrices = None
    if "gaussian_mixture" in formula_names and sigma_list is not None:
        sl = sigma_list if isinstance(sigma_list, list) else [sigma_list]
        mmd2, _, gauss_matrices = compute_gaussian_mmd2_stats(data, qfake, sl)
        v[name_to_idx["gaussian_mixture"]] = mmd2

    # For spatial_rect bootstrap: pre-compute rect parities for all samples
    spatial_rect_K_pp = spatial_rect_K_qq = spatial_rect_K_pq = None
    if "spatial_rect" in formula_names and grid_shape is not None:
        from src.core.metrics import _enumerate_rect_masks, _kernel_conv_parity
        H, W = grid_shape
        rects = _enumerate_rect_masks(nq, H, W, max_pw, max_ph)
        if len(rects) > 0:
            spatial_rect_K_pp = _kernel_conv_parity(data, data, rects)
            spatial_rect_K_qq = _kernel_conv_parity(qfake, qfake, rects)
            spatial_rect_K_pq = _kernel_conv_parity(data, qfake, rects)
            m2, _, _ = compute_spatial_rect_stats(data, qfake, grid_shape, max_pw, max_ph)
            v[name_to_idx["spatial_rect"]] = m2

    # ---- bootstrap Σ ----
    Z_data = {}
    Z_q = {}
    for name in geo_names:
        ops = kernel_masks.get(name)
        if ops is not None and ops.shape[0] > 0:
            Z_data[name] = 1 - 2 * ((data @ ops.T) % 2)
            Z_q[name] = 1 - 2 * ((qfake @ ops.T) % 2)

    B = n_bootstrap
    v_boot = np.zeros((B, M), dtype=np.float64)
    for b in range(B):
        idx_p = rng.choice(m, m, replace=True)
        idx_q = rng.choice(n_qf, n_qf, replace=True)

        for name in geo_names:
            if name not in Z_data:
                continue
            E_P = np.mean(Z_data[name][idx_p], axis=0)
            E_Q = np.mean(Z_q[name][idx_q], axis=0)
            v_boot[b, name_to_idx[name]] = float(np.sum((E_P - E_Q) ** 2))

        if "gaussian_mixture" in formula_names and gauss_matrices is not None:
            K_pp, K_qq, K_pq = gauss_matrices
            v_boot[b, name_to_idx["gaussian_mixture"]] = _bootstrap_gaussian_mmd2(
                K_pp, K_qq, K_pq, idx_p, idx_q)
        if "spatial_rect" in formula_names and spatial_rect_K_pp is not None:
            K_pp_r = spatial_rect_K_pp[np.ix_(idx_p, idx_p)]
            K_qq_r = spatial_rect_K_qq[np.ix_(idx_q, idx_q)]
            K_pq_r = spatial_rect_K_pq[np.ix_(idx_p, idx_q)]
            m_r = K_pp_r.shape[0]
            n_r = K_qq_r.shape[0]
            usp = (np.sum(K_pp_r) - m_r) / (m_r * (m_r - 1)) if m_r > 1 else 0.0
            usq = (np.sum(K_qq_r) - n_r) / (n_r * (n_r - 1)) if n_r > 1 else 0.0
            v_boot[b, name_to_idx["spatial_rect"]] = max(usp + usq - 2 * np.mean(K_pq_r), 0.0)

    Sigma = np.cov(v_boot, rowvar=False)
    # Clamp diagonal to prevent degenerate Σ
    diag_min = 1e-15
    for i in range(M):
        Sigma[i, i] = max(Sigma[i, i], diag_min)

    return v, Sigma


def optimize_mkl_weights(
    v: np.ndarray,
    Sigma: np.ndarray,
    lam: float = 1e-4,
) -> np.ndarray:
    """Solve α* = argmax_{α≥0, Σα=1}  αᵀv / √(αᵀΣα + λ‖α‖²).

    Uses scipy SLSQP with analytical gradient.
    """
    import scipy.optimize as opt

    M = len(v)

    def objective(alpha):
        alpha = np.asarray(alpha, dtype=np.float64)
        num = np.dot(alpha, v)
        denom = np.sqrt(np.dot(alpha, Sigma @ alpha) + lam * np.dot(alpha, alpha))
        if denom < 1e-30:
            return 0.0
        return -num / denom

    def gradient(alpha):
        alpha = np.asarray(alpha, dtype=np.float64)
        num = np.dot(alpha, v)
        denom_sq = np.dot(alpha, Sigma @ alpha) + lam * np.dot(alpha, alpha)
        denom = np.sqrt(max(denom_sq, 1e-30))

        grad_num = v
        grad_denom_sq = 2 * (Sigma @ alpha + lam * alpha)
        grad_denom = grad_denom_sq / (2 * max(denom, 1e-30))

        return -(grad_num * denom - num * grad_denom) / denom_sq

    constraints = [
        {'type': 'eq', 'fun': lambda a: np.sum(a) - 1.0},
    ]
    bounds = [(0.0, 1.0)] * M

    x0 = np.full(M, 1.0 / M)
    result = opt.minimize(
        objective, x0, method='SLSQP',
        jac=gradient,
        bounds=bounds, constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12},
    )

    alpha_star = result.x
    alpha_star = np.maximum(alpha_star, 0.0)
    alpha_star = alpha_star / max(alpha_star.sum(), 1e-30)
    return alpha_star


def run_mkl_pipeline(
    data: np.ndarray,
    grid_shape: tuple[int, int],
    base_kernels: list[str] | None = None,
    qfake_method: str = "scramble",
    qfake_noise_rate: float = 0.2,
    regularization: float = 1e-4,
    seed: int = 42,
    sigma_list: list[float] | None = None,
    max_pw: int = 4,
    max_ph: int = 4,
    n_bootstrap: int = 200,
) -> dict:
    """Run the full MKL pre-processing pipeline.

    Supports both geometric and formula-based kernels (gaussian_mixture,
    spatial_rect).

    Args:
        data: Training data (m, n_qubits) as 0/1 int8.
        grid_shape: (H, W) grid dimensions.
        base_kernels: List of kernel names to include. If None, uses all.
        qfake_method: Method to generate Q_fake.
        qfake_noise_rate: Noise rate for thermal Q_fake.
        regularization: λ for ‖α‖² ridge penalty.
        seed: Random seed.
        sigma_list: Sigma values for gaussian_mixture and spatial_rect
                    formula-based kernels.
        max_pw, max_ph: Max rect patch size for spatial_rect.
        n_bootstrap: Bootstrap iterations for covariance estimation.

    Returns dict with α*, per-kernel MMD² values, and metadata.
    """
    H, W = grid_shape
    if base_kernels is None:
        base_kernels = BASE_KERNEL_NAMES

    geo_kernels = [k for k in base_kernels if k not in _FORMULA_KERNEL_NAMES]
    kernel_masks = enumerate_all_base_kernels(H, W, names=geo_kernels)
    qfake = generate_q_fake(data, method=qfake_method, noise_rate=qfake_noise_rate, seed=seed)
    v, Sigma = compute_mkl_statistics(
        data, qfake, kernel_masks,
        all_base_kernels=base_kernels,
        sigma_list=sigma_list, grid_shape=grid_shape,
        max_pw=max_pw, max_ph=max_ph,
        n_bootstrap=n_bootstrap, seed=seed,
    )
    alpha = optimize_mkl_weights(v, Sigma, lam=regularization)

    return {
        "mkl_weights": alpha.tolist(),
        "mkl_base_kernels": base_kernels,
        "mmd2_per_kernel": v.tolist(),
        "grid_shape": [H, W],
        "qfake_method": qfake_method,
        "qfake_noise_rate": qfake_noise_rate,
        "regularization": regularization,
    }


# ---------------------------------------------------------------------------
# JIT-compatible MKL operator sampling (plugs into dual_mmd_loss)
# ---------------------------------------------------------------------------

_MKL_KERNEL_FNS = {}
for name in BASE_KERNEL_NAMES:
    _MKL_KERNEL_FNS[name] = name


def sample_base_kernel_ops(key, kernel_name: str, grid_shape: tuple,
                            n_ops: int, n_qubits: int, wires: list) -> tuple:
    """Sample operators from a single base kernel.

    For fully-enumerated kernels (density, nn_h, etc.), samples n_ops
    operators uniformly from their enumerated set.

    Returns (all_ops, visible_ops, op_weights_k).
    """
    H, W = grid_shape
    masks = enumerate_base_kernel_ops(kernel_name, H, W)
    n_avail = masks.shape[0]
    if n_avail == 0:
        return make_ops_from_pmf(
            key, jnp.ones(n_qubits + 1) / (n_qubits + 1),
            n_ops, n_qubits, wires, within_shell="shell_uniform",
        )

    indices = jax.random.randint(key, (n_ops,), 0, max(n_avail, 1))
    masks_jnp = jnp.asarray(masks, dtype=jnp.float64)

    if wires is None:
        wires = list(range(n_qubits))
    n_visible = len(wires)
    wire_indices = jnp.array(wires, dtype=int)

    all_ops = masks_jnp[indices]
    visible_ops = all_ops[:, wire_indices]
    op_weights_k = jnp.sum(visible_ops, axis=1).astype(int)

    return all_ops, visible_ops, op_weights_k


def build_mkl_ops(
    key: jax.Array,
    alphas: jnp.ndarray,
    base_kernel_names: list[str],
    grid_shape: tuple[int, int],
    n_ops: int,
    n_qubits: int,
    wires: list,
    gaussian_pmf: jnp.ndarray | None = None,
    max_pw: int = 4,
    max_ph: int = 4,
) -> tuple:
    """Sample n_ops operators from the α-weighted MKL dictionary.

    Supports geometric kernels (enumerated Pauli-Z masks), ``gaussian_mixture``
    (importance-sampled from Gaussian PMF), and ``spatial_rect`` (random rect
    patches with parity kernel).

    JIT-compatible: pre-samples n_ops operators from EVERY base kernel,
    then per-position picks via α-categorical one-hot gather. This avoids
    dynamic shapes (unlike per-kernel concatenation).

    Args:
        key: JAX PRNG key.
        alphas: (M,) mixture weights (must sum to 1).
        base_kernel_names: List of kernel names.
        grid_shape: (H, W) grid dimensions.
        n_ops: Number of operators to sample.
        n_qubits: Total qubits.
        wires: Visible wire indices.
        gaussian_pmf: PMF for gaussian_mixture (n_qubits+1,). Required if
                      ``"gaussian_mixture"`` is in base_kernel_names.
        max_pw, max_ph: Max rect patch size for spatial_rect kernel.

    Returns (all_ops, visible_ops, op_weights_k).
    """
    from src.core.importance_sampling import make_ops_from_pmf

    M = len(base_kernel_names)
    alphas = jnp.asarray(alphas, dtype=jnp.float64)
    alphas = alphas / jnp.maximum(jnp.sum(alphas), 1e-30)

    H, W = grid_shape
    wire_indices = jnp.array(wires if wires is not None else list(range(n_qubits)), dtype=int)
    n_visible = len(wire_indices)

    # Pre-sample n_ops operators from every kernel → (M, n_ops, n_qubits)
    all_kern_ops = []
    all_kern_vis = []
    all_kern_w = []

    for m in range(M):
        name = base_kernel_names[m]
        key_m, key = jax.random.split(key)

        if name == "gaussian_mixture":
            if gaussian_pmf is None:
                raise ValueError("gaussian_mixture requires gaussian_pmf")
            ops_m, vis_m, w_m = make_ops_from_pmf(
                key_m, gaussian_pmf, n_ops, n_qubits, wires,
                within_shell="shell_uniform",
            )
        elif name == "spatial_rect":
            masks_jnp = jnp.asarray(enumerate_spatial_rect_masks(H, W, max_pw, max_ph),
                                    dtype=jnp.float64)
            n_avail = masks_jnp.shape[0]
            if n_avail <= 1:
                idx_m = jnp.zeros(n_ops, dtype=jnp.int32)
            else:
                idx_m = jax.random.randint(key_m, (n_ops,), 0, n_avail)
            ops_m = masks_jnp[idx_m]
            vis_m = ops_m[:, wire_indices]
            w_m = jnp.sum(vis_m, axis=1).astype(jnp.int32)
        else:
            masks = enumerate_base_kernel_ops(name, H, W)
            masks_jnp = jnp.asarray(masks, dtype=jnp.float64)
            n_avail = masks_jnp.shape[0]
            if n_avail <= 1:
                idx_m = jnp.zeros(n_ops, dtype=jnp.int32)
            else:
                idx_m = jax.random.randint(key_m, (n_ops,), 0, n_avail)
            ops_m = masks_jnp[idx_m]
            vis_m = ops_m[:, wire_indices]
            w_m = jnp.sum(vis_m, axis=1).astype(jnp.int32)

        all_kern_ops.append(ops_m)
        all_kern_vis.append(vis_m)
        all_kern_w.append(w_m)

    stack_ops = jnp.stack(all_kern_ops, axis=0)    # (M, n_ops, n_qubits)
    stack_vis = jnp.stack(all_kern_vis, axis=0)    # (M, n_ops, n_visible)
    stack_w = jnp.stack(all_kern_w, axis=0)        # (M, n_ops)

    # α-categorical choice per operator position
    k_choice, key = jax.random.split(key)
    logits = jnp.log(jnp.maximum(alphas, 1e-30))
    choices = jax.random.categorical(k_choice, logits, shape=(n_ops,))  # (n_ops,)

    # One-hot gather
    onehot = jax.nn.one_hot(choices, M, dtype=jnp.float64)  # (n_ops, M)
    all_ops = jnp.einsum('pk,kpq->pq', onehot, stack_ops)      # (n_ops, n_qubits)
    visible_ops = jnp.einsum('pk,kpq->pq', onehot, stack_vis)   # (n_ops, n_visible)
    op_weights_k = jnp.einsum('pk,kp->p', onehot, stack_w).astype(jnp.int32)

    return all_ops, visible_ops, op_weights_k


def build_mkl_ops_via_pmf(
    key: jax.Array,
    alpha_pmfs: list[jnp.ndarray],
    base_kernel_names: list[str],
    grid_shape: tuple[int, int],
    n_ops: int,
    n_qubits: int,
    wires: list,
) -> tuple:
    """Alternative MKL sampling via α-weighted PMF mix.

    Computes the composite PMF P_mix = Σ α_m · P_m where P_m is the
    weight-k PMF for the m-th base kernel, then samples from it via
    make_ops_from_pmf. This works only for base kernels that can be
    fully described by a Hamming-weight PMF.

    For geometrically structured kernels, use build_mkl_ops instead.
    """
    M = len(base_kernel_names)
    alphas = jnp.asarray(alpha_pmfs, dtype=jnp.float64)
    if alphas.ndim == 1:
        alphas = alphas[None, :]
    composite = jnp.sum(alphas, axis=0)
    composite = composite / jnp.maximum(jnp.sum(composite), 1e-30)
    return make_ops_from_pmf(
        key, composite, n_ops, n_qubits, wires,
        within_shell="shell_uniform",
    )
