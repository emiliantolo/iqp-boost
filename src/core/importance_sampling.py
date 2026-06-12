"""Modular Importance Sampling functions for ensemble boosting."""

import numpy as np
from math import comb
import jax
import jax.numpy as jnp


def is_pmf_like(sigma) -> bool:
    """Check if sigma is an array-like PMF (length n+1 over Hamming weights)."""
    if isinstance(sigma, (int, float)):
        return False
    if not hasattr(sigma, '__len__'):
        return False
    return True


def sigma_to_binomial_pmf(sigma: float, n_visible: int) -> np.ndarray:
    """Convert a scalar Gaussian kernel bandwidth (sigma) into its equivalent Binomial PMF over Hamming weights."""
    p = (1.0 - np.exp(-1.0 / (2.0 * float(sigma)**2))) / 2.0
    pmf = np.zeros(n_visible + 1, dtype=np.float64)
    for k in range(n_visible + 1):
        pmf[k] = comb(n_visible, k) * (p**k) * ((1.0 - p)**(n_visible - k))
    return pmf


def _resolve_pmfs(sigma, n_visible: int) -> list[np.ndarray]:
    """Resolve ensemble.sigma (scalar, list of scalars, PMF, or list of PMFs) into a list of length-(n_visible+1) PMFs.

    For dict sigma of type ``"spatial_gaussian_mixture"``, returns a single
    entry that is the composite of all Gaussian bandwidth PMFs (averaged).

    Returns a list of np.ndarray PMFs, one per bandwidth.
    """
    if isinstance(sigma, dict) and sigma.get('type') == 'spatial_gaussian_mixture':
        gauss_pmfs = _resolve_pmfs(sigma['sigma'], n_visible)
        composite = np.mean(gauss_pmfs, axis=0)
        composite = composite / max(composite.sum(), 1e-30)
        return [composite]
    if isinstance(sigma, dict) and sigma.get('type') == 'mkl':
        gauss_sigma = sigma.get('sigma', None)
        if gauss_sigma is not None:
            return _resolve_pmfs(gauss_sigma, n_visible)
        return [sigma_to_binomial_pmf(1.0, n_visible)]
    if isinstance(sigma, (int, float)):
        return [sigma_to_binomial_pmf(float(sigma), n_visible)]
    if not hasattr(sigma, '__len__'):
        raise ValueError(f"sigma must be scalar or array-like, got {type(sigma)}")

    arr = np.asarray(sigma, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == n_visible + 1:
        pmf = arr / max(arr.sum(), 1e-30)
        return [pmf]
    if arr.ndim == 1:
        return [sigma_to_binomial_pmf(float(s), n_visible) for s in arr]
    raise ValueError(f"sigma must be scalar, list of scalars, or PMF, got shape {arr.shape}")


def _sample_shells_uniform(key, k_array: jnp.ndarray, n_visible: int) -> jnp.ndarray:
    """Sample n_ops weight-k_i vectors uniformly from the shell on {0,...,n_visible-1}.

    Args:
        key: JAX PRNG key
        k_array: (n_ops,) integer array of Hamming weights
        n_visible: number of qubits in the visible set

    Returns:
        (n_ops, n_visible) binary matrix (float64)
    """
    n_ops = k_array.shape[0]
    g = jax.random.gumbel(key, (n_ops, n_visible))
    sorted_idx = jnp.argsort(-g, axis=1)
    pos = jnp.arange(n_visible)
    keep = (pos[None, :] < k_array[:, None]).astype(jnp.int32)
    one_hot = jax.nn.one_hot(sorted_idx, n_visible, dtype=jnp.int32)
    result = jnp.einsum('ij,ijn->in', keep, one_hot)
    return result.astype(jnp.float64)


def make_ops_from_pmf(key, P, n_ops: int, n_qubits: int, wires: list, within_shell: str = "shell_uniform") -> tuple:
    """Sample n_ops operators from the PMF P over Hamming weight classes.

    Args:
        key: JAX PRNG key
        P: length-(n_visible+1) PMF over Hamming weights
        n_ops: number of operators to sample
        n_qubits: total number of qubits (including ancillas)
        wires: list of visible qubit indices
        within_shell: "bernoulli" for per-bit independent draw (backward compat);
                      "shell_uniform" for sample k ~ P then uniform within shell.

    Returns:
        (all_ops, visible_ops, op_weights_k) where:
            all_ops: (n_ops, n_qubits) binary matrix
            visible_ops: (n_ops, n_visible) binary matrix
            op_weights_k: (n_ops,) integer array of Hamming weights
    """
    if wires is None:
        wires = list(range(n_qubits))
    n_visible = len(wires)
    P = jnp.asarray(P, dtype=jnp.float64)
    P = P / jnp.maximum(P.sum(), 1e-30)

    if within_shell == "bernoulli":
        p = jnp.sum(jnp.arange(n_visible + 1, dtype=jnp.float64) * P) / n_visible
        visible_ops = jax.random.bernoulli(key, p, shape=(n_ops, n_visible)).astype(jnp.float64)
    elif within_shell == "shell_uniform":
        k_array = jax.random.choice(key, n_visible + 1, shape=(n_ops,), p=P)
        visible_ops = _sample_shells_uniform(key, k_array, n_visible)
    else:
        raise ValueError(f"Unknown within_shell: {within_shell}. Use 'bernoulli' or 'shell_uniform'.")

    all_ops = jnp.zeros((n_ops, n_qubits), dtype=jnp.float64)
    wire_indices = jnp.array(wires, dtype=int)
    all_ops = all_ops.at[:, wire_indices].set(visible_ops)

    op_weights_k = jnp.sum(visible_ops, axis=1).astype(int)
    return all_ops, visible_ops, op_weights_k


def make_ops_uniform(key, P, n_ops: int, n_qubits: int, wires: list) -> tuple:
    """Convenience wrapper for shell-uniform sampling (the new default)."""
    return make_ops_from_pmf(key, P, n_ops, n_qubits, wires, within_shell="shell_uniform")


def compute_per_op_ratios(P_global, Q_t, op_weights_k) -> jnp.ndarray:
    """Compute per-op importance ratio P_global[k(omega)] / Q_t[k(omega)].

    Args:
        P_global: length-(n_visible+1) target PMF
        Q_t: length-(n_visible+1) proposal PMF
        op_weights_k: (n_ops,) integer array of Hamming weights per operator

    Returns:
        (n_ops,) float array of importance ratios
    """
    P_global = jnp.asarray(P_global, dtype=jnp.float64)
    Q_t = jnp.asarray(Q_t, dtype=jnp.float64)
    safe_Q = jnp.maximum(Q_t, 1e-12)
    safe_Q = safe_Q / safe_Q.sum()
    return P_global[op_weights_k] / safe_Q[op_weights_k]


def compute_dynamic_proposal(ensemble, x_train: np.ndarray, step: int, config: dict, key: jax.Array = None) -> tuple:
    """Compute the dynamically focused proposal distributions Q_t and target PMFs P_global.

    Returns:
        (list_Q_t, list_P_global) as parallel lists of length n_sigmas, each entry
        a length-(n_visible+1) numpy array.
    """
    n_visible = len(ensemble.wires) if ensemble.wires is not None else ensemble.iqp_circuit.n_qubits
    list_P_global = _resolve_pmfs(ensemble.sigma, n_visible)

    if step == 0 or not ensemble.models:
        return list(list_P_global), list(list_P_global)

    if not ensemble.terms.trs or not ensemble.terms.ops:
        return list(list_P_global), list(list_P_global)

    list_Q_t = []
    beta = float(config.get('dynamic_is_beta', 0.05))

    for sigma_idx, P_global in enumerate(list_P_global):
        if sigma_idx not in ensemble.terms.ops:
            list_Q_t.append(P_global.copy())
            continue

        all_ops_prev, visible_ops_prev = ensemble.terms.ops[sigma_idx]
        try:
            tr_enss = jnp.array([t[sigma_idx] for t in ensemble.terms.trs])
        except (IndexError, TypeError):
            list_Q_t.append(P_global.copy())
            continue

        D_j = jnp.mean(1 - 2 * ((x_train @ visible_ops_prev.T) % 2), axis=0)
        w = jnp.array(ensemble.weights)
        E_j = jnp.sum(w[:, None] * tr_enss, axis=0) if tr_enss.shape[0] > 0 else jnp.zeros(visible_ops_prev.shape[0])

        residuals_sq = (E_j - D_j) ** 2
        op_weights_k = jnp.sum(visible_ops_prev, axis=1).astype(int)

        E_bar = np.zeros(n_visible + 1)
        for k in range(n_visible + 1):
            mask = (op_weights_k == k)
            if jnp.any(mask):
                E_bar[k] = jnp.mean(residuals_sq[mask])

        P_focus = P_global * E_bar
        if P_focus.sum() > 0:
            P_focus = P_focus / P_focus.sum()
        else:
            P_focus = P_global.copy()

        Q_t = (1.0 - beta) * P_focus + beta * P_global
        Q_t = Q_t / Q_t.sum()
        list_Q_t.append(Q_t)

    return list_Q_t, list_P_global


def _sample_rect_mask(cx: int, cy: int, w: int, h: int,
                       H: int, W: int) -> np.ndarray:
    """Build a 2D (H,W) mask: 1 inside the centered odd-sized rectangle, 0 elsewhere."""
    half_w = (w - 1) // 2
    half_h = (h - 1) // 2
    r0 = max(cy - half_h, 0)
    r1 = min(cy + half_h + 1, H)
    c0 = max(cx - half_w, 0)
    c1 = min(cx + half_w + 1, W)
    mask = np.zeros((H, W), dtype=np.float64)
    mask[r0:r1, c0:c1] = 1.0
    return mask


def _spatial_rect_visible(key, lam: float, grid_shape: tuple,
                           max_pw: int, max_ph: int,
                           n_ops: int, n_visible: int,
                           gaussian_pmf: jnp.ndarray,
                           n_qubits: int, wires: list):
    """Parity kernel: hard 1s inside the centered odd rectangle.

    Each spatial op is a hard rectangle mask (all 1s inside patch, 0 outside).
    This corresponds to K_conv(x,y) = E_rect[(-1)^{popcount(x⊕y within rect)}].
    """
    H, W = grid_shape
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    is_spatial = jax.random.bernoulli(k1, lam, shape=(n_ops,))

    odd_ws = jnp.array([w for w in range(1, max_pw + 1, 2)])
    odd_hs = jnp.array([h for h in range(1, max_ph + 1, 2)])

    wi = jax.random.randint(k2, (n_ops,), 0, len(odd_ws))
    hi = jax.random.randint(k3, (n_ops,), 0, len(odd_hs))
    widths = odd_ws[wi]
    heights = odd_hs[hi]
    half_w = (widths - 1) // 2
    half_h = (heights - 1) // 2

    cx = jax.random.randint(k4, (n_ops,), 0, W)
    cy = jax.random.randint(k5, (n_ops,), 0, H)
    cx = jnp.clip(cx, half_w, W - 1 - half_w)
    cy = jnp.clip(cy, half_h, H - 1 - half_h)

    wire_rows = jnp.arange(n_visible) // W
    wire_cols = jnp.arange(n_visible) % W

    r0 = cy[:, None] - half_h[:, None]
    r1 = cy[:, None] + half_h[:, None] + 1
    c0 = cx[:, None] - half_w[:, None]
    c1 = cx[:, None] + half_w[:, None] + 1

    in_rect = (wire_rows[None, :] >= r0) & (wire_rows[None, :] < r1) \
            & (wire_cols[None, :] >= c0) & (wire_cols[None, :] < c1)
    spatial_visible = in_rect.astype(jnp.float64)

    _, gauss_visible, _ = make_ops_from_pmf(
        k1, gaussian_pmf, n_ops, n_qubits, wires,
        within_shell="shell_uniform",
    )

    mask = is_spatial[:, None].astype(jnp.float64)
    visible_ops = mask * spatial_visible + (1.0 - mask) * gauss_visible
    op_weights_k = jnp.sum(visible_ops, axis=1).astype(int)

    all_ops = jnp.zeros((n_ops, n_qubits), dtype=jnp.float64)
    wire_indices = jnp.array(wires, dtype=int)
    all_ops = all_ops.at[:, wire_indices].set(visible_ops)

    return all_ops, visible_ops, op_weights_k


def _sample_patch_binomial(key, lam: float, grid_shape: tuple,
                            max_pw: int, max_ph: int,
                            n_ops: int, n_visible: int,
                            gaussian_pmf: jnp.ndarray,
                            n_qubits: int, wires: list):
    """Gaussian kernel: independent Bernoulli(p) inside the centered odd patch.

    Each spatial op flips independent coins at rate p inside the patch,
    giving K_conv(x,y) = E_rect[exp(-d_H(x_P,y_P) / 2σ²)].
    p = mean(Hamming weight under composite PMF) / n_visible.
    """
    H, W = grid_shape
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    k_vals = jnp.arange(n_visible + 1, dtype=jnp.float64)
    p_eff = jnp.sum(k_vals * gaussian_pmf) / n_visible
    p_eff = jnp.clip(p_eff, 1e-8, 1.0 - 1e-8)

    is_spatial = jax.random.bernoulli(k1, lam, shape=(n_ops,))

    odd_ws = jnp.array([w for w in range(1, max_pw + 1, 2)])
    odd_hs = jnp.array([h for h in range(1, max_ph + 1, 2)])

    wi = jax.random.randint(k2, (n_ops,), 0, len(odd_ws))
    hi = jax.random.randint(k3, (n_ops,), 0, len(odd_hs))
    widths = odd_ws[wi]
    heights = odd_hs[hi]
    half_w = (widths - 1) // 2
    half_h = (heights - 1) // 2

    cx = jax.random.randint(k4, (n_ops,), 0, W)
    cy = jax.random.randint(k5, (n_ops,), 0, H)
    cx = jnp.clip(cx, half_w, W - 1 - half_w)
    cy = jnp.clip(cy, half_h, H - 1 - half_h)

    wire_rows = jnp.arange(n_visible) // W
    wire_cols = jnp.arange(n_visible) % W

    r0 = cy[:, None] - half_h[:, None]
    r1 = cy[:, None] + half_h[:, None] + 1
    c0 = cx[:, None] - half_w[:, None]
    c1 = cx[:, None] + half_w[:, None] + 1

    in_rect = (wire_rows[None, :] >= r0) & (wire_rows[None, :] < r1) \
            & (wire_cols[None, :] >= c0) & (wire_cols[None, :] < c1)

    inside_key = jax.random.split(k6)[0]
    inside_coins = jax.random.uniform(inside_key, (n_ops, n_visible))
    coins = jnp.where(in_rect, inside_coins, 1.0)
    spatial_visible = (coins < p_eff).astype(jnp.float64)

    _, gauss_visible, _ = make_ops_from_pmf(
        k1, gaussian_pmf, n_ops, n_qubits, wires,
        within_shell="shell_uniform",
    )

    mask = is_spatial[:, None].astype(jnp.float64)
    visible_ops = mask * spatial_visible + (1.0 - mask) * gauss_visible
    op_weights_k = jnp.sum(visible_ops, axis=1).astype(int)

    all_ops = jnp.zeros((n_ops, n_qubits), dtype=jnp.float64)
    wire_indices = jnp.array(wires, dtype=int)
    all_ops = all_ops.at[:, wire_indices].set(visible_ops)

    return all_ops, visible_ops, op_weights_k


_SPATIAL_KERNELS = {
    "parity": _spatial_rect_visible,
    "gaussian": _sample_patch_binomial,
}


def build_spatial_mixture_ops(
    key: jax.Array,
    gaussian_pmf: np.ndarray,
    lam: float,
    grid_shape: tuple[int, int],
    max_pw: int, max_ph: int,
    n_ops: int, n_qubits: int, wires: list,
    kernel: str = "parity",
) -> tuple:
    """Sample operators from the convex mixture:
        w^mix = λ · w^conv + (1-λ) · w^gauss

    w^conv is defined by the ``kernel`` parameter:
      - ``"parity"`` (default): hard 1s inside the patch → K = (-1)^{popcount}
      - ``"gaussian"``: independent Bernoulli(p) inside the patch → K = exp(-d_H / 2σ²)

    Uses per-op Bernoulli blending so shapes are static and the function
    is JIT-compatible.

    Returns (all_ops, visible_ops, op_weights_k) matching
    ``make_ops_from_pmf``.
    """
    n_visible = len(wires)
    gaussian_pmf = jnp.asarray(gaussian_pmf, dtype=jnp.float64)
    fn = _SPATIAL_KERNELS.get(kernel)
    if fn is None:
        raise ValueError(f"Unknown spatial kernel: {kernel}. Use 'parity' or 'gaussian'.")
    return fn(
        key, float(lam), grid_shape, int(max_pw), int(max_ph),
        int(n_ops), int(n_visible), gaussian_pmf, int(n_qubits), wires,
    )


def populate_traces_for_is(ensemble, key: jax.Array, config: dict) -> None:
    """Refresh EnsembleTerms cache for dynamic IS at the end of training.

    Re-samples operators from P_global (not Q_{t-1}, to avoid recursive reweighting),
    re-evaluates every model on the new operators, and stores fresh traces / corrections
    / op_weights_k. This makes Q_t at the start of the next step read a fresh, consistent cache.
    """
    n_visible = len(ensemble.wires) if ensemble.wires is not None else ensemble.iqp_circuit.n_qubits
    list_P_global = _resolve_pmfs(ensemble.sigma, n_visible)
    within_shell = config.get('dynamic_is_within_shell', 'shell_uniform')
    n_samples = ensemble.n_samples
    max_batch_ops = ensemble.max_batch_ops
    max_batch_samples = ensemble.max_batch_samples

    ensemble.terms.trs.clear()
    ensemble.terms.corrs.clear()
    ensemble.terms.sample_ops(
        ensemble.iqp_circuit, ensemble.sigma, ensemble.n_ops, key,
        wires=ensemble.wires, pmfs=list_P_global, within_shell=within_shell
    )

    for m_params in ensemble.models:
        key, subkey = jax.random.split(key, 2)
        ensemble.terms.add_term(
            m_params, ensemble.iqp_circuit, ensemble.sigma, ensemble.n_ops,
            n_samples, subkey, wires=ensemble.wires,
            max_batch_ops=max_batch_ops, max_batch_samples=max_batch_samples
        )
