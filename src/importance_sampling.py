"""Modular Importance Sampling functions for ensemble boosting."""

import numpy as np
from math import comb
import jax
import jax.numpy as jnp

def is_pmf_like(sigma) -> bool:
    """Check if sigma is an array-like PMF."""
    return isinstance(sigma, (list, tuple, np.ndarray, jnp.ndarray))

def sigma_to_binomial_pmf(sigma: float, n_qubits: int) -> np.ndarray:
    """Convert a scalar Gaussian kernel bandwidth (sigma) into its equivalent Binomial PMF over Hamming weights."""
    p = (1.0 - np.exp(-1.0 / (2.0 * float(sigma)**2))) / 2.0
    pmf = np.zeros(n_qubits + 1, dtype=np.float64)
    for k in range(n_qubits + 1):
        pmf[k] = comb(n_qubits, k) * (p**k) * ((1.0 - p)**(n_qubits - k))
    return pmf

def compute_dynamic_proposal(ensemble, x_train: np.ndarray, step: int, config: dict, key: jax.Array = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute the dynamically focused proposal distribution Q_t and global target PMF P_global.

    Returns:
        (Q_t, P_global) as 1D numpy arrays of length n_qubits + 1.
    """
    n_qubits = len(ensemble.wires) if ensemble.wires is not None else ensemble.iqp_circuit.n_qubits

    # 1. Establish P_global
    if is_pmf_like(ensemble.sigma):
        P_global = np.asarray(ensemble.sigma, dtype=np.float64)
    else:
        P_global = sigma_to_binomial_pmf(ensemble.sigma, n_qubits)

    # Step 0: proposal is simply target distribution
    if step == 0 or not ensemble.models:
        return P_global, P_global

    # 2. Extract or compute residuals from step t-1
    if ensemble.terms.ops and ensemble.terms.trs:
        # Caching is active: reuse cached operators and traces
        all_ops_prev, visible_ops_prev = ensemble.terms.ops[0]
        tr_enss = jnp.array([t[0] for t in ensemble.terms.trs])
    else:
        # Caching is disabled: dynamically sample and evaluate previous model traces
        from src.dual_mmd_loss import _make_ops
        if key is None:
            key = jax.random.PRNGKey(config.get('rng_seed', 0) + step * 7919)
        key, temp_key = jax.random.split(key)
        all_ops_prev, visible_ops_prev = _make_ops(
            temp_key, P_global, ensemble.n_ops, ensemble.iqp_circuit.n_qubits, ensemble.wires
        )
        
        tr_enss_list = []
        for m_params in ensemble.models:
            key, subkey = jax.random.split(key)
            samples = ensemble.iqp_circuit.op_expval(
                m_params, all_ops_prev, ensemble.n_samples, subkey,
                indep_estimates=config.get('indep_estimates', False),
                return_samples=True, max_batch_ops=config.get('max_batch_ops', None),
                max_batch_samples=config.get('max_batch_samples', None)
            )
            tr_enss_list.append(jnp.mean(samples, axis=-1))
        tr_enss = jnp.array(tr_enss_list) if tr_enss_list else jnp.zeros((0, ensemble.n_ops))

    D_j = jnp.mean(1 - 2 * ((x_train @ visible_ops_prev.T) % 2), axis=0)
    w = jnp.array(ensemble.weights)
    E_j = jnp.sum(w[:, None] * tr_enss, axis=0) if len(tr_enss) > 0 else jnp.zeros(ensemble.n_ops)

    residuals_sq = (E_j - D_j)**2
    op_weights_k = jnp.sum(visible_ops_prev, axis=1).astype(int)

    # 3. Calculate mean residual error per Hamming weight shell
    E_bar = np.zeros(n_qubits + 1)
    for k in range(n_qubits + 1):
        mask = (op_weights_k == k)
        if jnp.any(mask):
            E_bar[k] = jnp.mean(residuals_sq[mask])

    # 4. Construct focus distribution
    P_focus = P_global * E_bar
    if P_focus.sum() > 0:
        P_focus /= P_focus.sum()
    else:
        P_focus = P_global.copy()

    # 5. Mix with target PMF to guarantee strict positivity on support of P_global
    beta = config.get('dynamic_is_beta', 0.05)
    Q_t = (1 - beta) * P_focus + beta * P_global
    Q_t /= Q_t.sum()

    return Q_t, P_global
