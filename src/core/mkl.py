"""Multiple Kernel Learning (MKL) for spatial-aware MMD.

Provides:
1. Geometric operator enumeration for the kernel dictionary
2. JIT-compatible MKL operator sampling (build_mkl_ops)
3. Standalone MKL optimization (maximise MMD test power)

Each base kernel is defined by an operator set S_m — a specific subset of
Pauli-Z strings on a 2D grid. The total kernel is K_alpha = sum alpha_m K_m
and MMD^2_alpha = sum alpha_m MMD^2_m.

The alpha* weights are found by maximising the Sutherland test statistic:
  alpha* = argmax_{alpha>=0}  alpha^T v / sqrt(alpha^T Sigma alpha + lambda||alpha||^2)
where v_m = MMD^2_m(P_data, Q_fake) and Sigma is the covariance of MMD^2 estimates.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from src.core.importance_sampling import make_ops_from_pmf

BASE_KERNEL_NAMES = [
    "density",          # 1-body: Z_i for every pixel
    "nn_h",             # 2-body: Z_i Z_j for horizontal nearest neighbours
    "nn_v",             # 2-body: Z_i Z_j for vertical nearest neighbours
    "nn_d",             # 2-body: Z_i Z_j for diagonal neighbours (\ and /)
    "plaquette",        # 4-body: Z_i Z_j Z_k Z_l per 2x2 block
    "global_anchor",    # N-body: row parities + column parities + full parity
]


# ---------------------------------------------------------------------------
# Geometric operator enumeration
# ---------------------------------------------------------------------------

def enumerate_density_ops(H: int, W: int) -> np.ndarray:
    n = H * W
    return np.eye(n, dtype=np.uint8)


def enumerate_nn_horizontal_ops(H: int, W: int) -> np.ndarray:
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
    n = H * W
    ops = []
    for i in range(H - 1):
        for j in range(W):
            mask = np.zeros(n, dtype=np.uint8)
            mask[i * W + j] = 1
            mask[(i + 1) * W + j] = 1
            ops.append(mask)
    return np.stack(ops) if ops else np.zeros((0, n), dtype=np.uint8)


def enumerate_nn_diagonal_ops(H: int, W: int) -> np.ndarray:
    n = H * W
    ops = []
    for i in range(H - 1):
        for j in range(W - 1):
            tl = i * W + j
            tr = i * W + (j + 1)
            bl = (i + 1) * W + j
            br = (i + 1) * W + (j + 1)
            m1 = np.zeros(n, dtype=np.uint8)
            m1[tl] = 1
            m1[br] = 1
            ops.append(m1)
            m2 = np.zeros(n, dtype=np.uint8)
            m2[tr] = 1
            m2[bl] = 1
            ops.append(m2)
    return np.stack(ops) if ops else np.zeros((0, n), dtype=np.uint8)


def enumerate_plaquette_ops(H: int, W: int) -> np.ndarray:
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


def enumerate_global_anchor_ops(H: int, W: int) -> np.ndarray:
    """Row parities (H) + column parities (W) + full parity (1)."""
    n = H * W
    ops = []
    for r in range(H):
        mask = np.zeros(n, dtype=np.uint8)
        for c in range(W):
            mask[r * W + c] = 1
        ops.append(mask)
    for c in range(W):
        mask = np.zeros(n, dtype=np.uint8)
        for r in range(H):
            mask[r * W + c] = 1
        ops.append(mask)
    ops.append(np.ones(n, dtype=np.uint8))
    return np.stack(ops) if ops else np.zeros((0, n), dtype=np.uint8)


_GEOMETRIC_KERNEL_NAMES = list(BASE_KERNEL_NAMES)

_ENUMERATORS = {
    "density": enumerate_density_ops,
    "nn_h": enumerate_nn_horizontal_ops,
    "nn_v": enumerate_nn_vertical_ops,
    "nn_d": enumerate_nn_diagonal_ops,
    "plaquette": enumerate_plaquette_ops,
    "global_anchor": enumerate_global_anchor_ops,
}


def enumerate_base_kernel_ops(kernel_name: str, H: int, W: int) -> np.ndarray:
    fn = _ENUMERATORS.get(kernel_name)
    if fn is None:
        raise ValueError(
            f"Unknown kernel '{kernel_name}'. "
            f"Available: {list(_ENUMERATORS)}."
        )
    return fn(H, W)


def enumerate_all_base_kernels(H: int, W: int, names: list[str] | None = None) -> dict[str, np.ndarray]:
    if names is None:
        names = BASE_KERNEL_NAMES
    return {name: enumerate_base_kernel_ops(name, H, W) for name in names}


# ---------------------------------------------------------------------------
# Expectation computation
# ---------------------------------------------------------------------------

def compute_operator_expectations(data: np.ndarray, ops: np.ndarray) -> np.ndarray:
    return np.mean(1 - 2 * ((data @ ops.T) % 2), axis=0)


def compute_mmd2_from_expectations(E_P: np.ndarray, E_Q: np.ndarray) -> float:
    return float(np.sum((E_P - E_Q) ** 2))


def compute_mmd2_variance(E_P: np.ndarray, E_Q: np.ndarray,
                          data: np.ndarray, qfake: np.ndarray) -> float:
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
# MKL optimisation
# ---------------------------------------------------------------------------

def generate_q_fake(data: np.ndarray, method: str = "scramble",
                    noise_rate: float = 0.2, seed: int = 42) -> np.ndarray:
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
    n_bootstrap: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    m, nq = data.shape
    n_qf = len(qfake)
    rng = np.random.default_rng(seed)

    names = list(kernel_masks.keys())
    M = len(names)
    name_to_idx = {n: i for i, n in enumerate(names)}

    v = np.zeros(M, dtype=np.float64)
    for name in names:
        ops = kernel_masks.get(name)
        if ops is not None and ops.shape[0] > 0:
            E_P = compute_operator_expectations(data, ops)
            E_Q = compute_operator_expectations(qfake, ops)
            v[name_to_idx[name]] = compute_mmd2_from_expectations(E_P, E_Q)

    Z_data = {}
    Z_q = {}
    for name in names:
        ops = kernel_masks.get(name)
        if ops is not None and ops.shape[0] > 0:
            Z_data[name] = 1 - 2 * ((data @ ops.T) % 2)
            Z_q[name] = 1 - 2 * ((qfake @ ops.T) % 2)

    B = n_bootstrap
    v_boot = np.zeros((B, M), dtype=np.float64)
    for b in range(B):
        idx_p = rng.choice(m, m, replace=True)
        idx_q = rng.choice(n_qf, n_qf, replace=True)

        for name in names:
            if name not in Z_data:
                continue
            E_P = np.mean(Z_data[name][idx_p], axis=0)
            E_Q = np.mean(Z_q[name][idx_q], axis=0)
            v_boot[b, name_to_idx[name]] = float(np.sum((E_P - E_Q) ** 2))

    Sigma = np.cov(v_boot, rowvar=False)
    diag_min = 1e-15
    for i in range(M):
        Sigma[i, i] = max(Sigma[i, i], diag_min)

    return v, Sigma


def optimize_mkl_weights(
    v: np.ndarray,
    Sigma: np.ndarray,
    lam: float = 1e-4,
) -> np.ndarray:
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

    constraints = [{'type': 'eq', 'fun': lambda a: np.sum(a) - 1.0}]
    bounds = [(0.0, 1.0)] * M
    x0 = np.full(M, 1.0 / M)
    result = opt.minimize(
        objective, x0, method='SLSQP', jac=gradient,
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
    n_bootstrap: int = 200,
) -> dict:
    H, W = grid_shape
    if base_kernels is None:
        base_kernels = BASE_KERNEL_NAMES

    kernel_masks = enumerate_all_base_kernels(H, W, names=base_kernels)
    qfake = generate_q_fake(data, method=qfake_method, noise_rate=qfake_noise_rate, seed=seed)
    v, Sigma = compute_mkl_statistics(data, qfake, kernel_masks, n_bootstrap=n_bootstrap, seed=seed)
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

def sample_base_kernel_ops(key, kernel_name: str, grid_shape: tuple,
                            n_ops: int, n_qubits: int, wires: list) -> tuple:
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
) -> tuple:
    H, W = grid_shape
    M = len(base_kernel_names)
    alphas = jnp.asarray(alphas, dtype=jnp.float64)
    alphas = alphas / jnp.maximum(jnp.sum(alphas), 1e-30)

    wire_indices = jnp.array(wires if wires is not None else list(range(n_qubits)), dtype=int)
    n_visible = len(wire_indices)

    all_kern_ops = []
    all_kern_vis = []
    all_kern_w = []

    for m in range(M):
        name = base_kernel_names[m]
        key_m, key = jax.random.split(key)

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

    stack_ops = jnp.stack(all_kern_ops, axis=0)
    stack_vis = jnp.stack(all_kern_vis, axis=0)
    stack_w = jnp.stack(all_kern_w, axis=0)

    k_choice, key = jax.random.split(key)
    logits = jnp.log(jnp.maximum(alphas, 1e-30))
    choices = jax.random.categorical(k_choice, logits, shape=(n_ops,))

    onehot = jax.nn.one_hot(choices, M, dtype=jnp.float64)
    all_ops = jnp.einsum('pk,kpq->pq', onehot, stack_ops)
    visible_ops = jnp.einsum('pk,kpq->pq', onehot, stack_vis)
    op_weights_k = jnp.einsum('pk,kp->p', onehot, stack_w).astype(jnp.int32)

    return all_ops, visible_ops, op_weights_k
