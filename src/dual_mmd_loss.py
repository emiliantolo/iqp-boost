"""Dual MMD loss for functional gradient boosting with Gaussian kernels.

Follows the same code style as iqpopt.gen_qml.iqp_methods:
  - dual_loss_estimate_iqp  (inner, per-sigma, JIT-friendly)
  - dual_mmd_loss           (outer, iterates over sigmas)

Key difference from mmd_loss_iqp: the dual loss requires shared operators between
ensemble traces and the new model. Operators are stored in EnsembleTerms alongside
the traces and reused across training epochs.
"""
from jax._src.typing import Array
import jax.numpy as jnp
import jax
import functools

from iqpopt import IqpSimulator


def _make_ops(key: Array, sigma: float, n_ops: int, n_qubits: int, wires: list):
    """Sample Gaussian kernel operators. Returns (all_ops, visible_ops)."""
    p_MMD = (1 - jnp.exp(-1 / 2 / sigma**2)) / 2
    visible_ops = jnp.array(jax.random.binomial(key, 1, p_MMD, shape=(n_ops, len(wires))), dtype='float64')

    all_ops = []
    i = 0
    for q in range(n_qubits):
        if q in wires:
            all_ops.append(visible_ops[:, i])
            i += 1
        else:
            all_ops.append(jnp.zeros(n_ops))
    all_ops = jnp.array(all_ops, dtype='float64').T
    return all_ops, visible_ops


def dual_loss_estimate_iqp(params: jnp.ndarray, iqp_circuit: IqpSimulator, ground_truth: jnp.ndarray,
                           visible_ops: jnp.ndarray, all_ops: jnp.ndarray,
                           tr_ens: jnp.ndarray, term_ens_ens: float,
                           n_samples: int, key: Array, lambda_dual: float = 1.0,
                           init_coefs: jnp.ndarray = None,
                           indep_estimates: bool = False,
                           return_components: bool = False,
                           max_batch_ops: int = None,
                           max_batch_samples: int = None) -> float | dict:
    """Estimates the dual MMD loss of new model P against data D and ensemble E.

    Loss = (1-lambda_dual) P^2 + 2(lambda_dual PE - PD) + (DD - lambda_dual EE)

    When lambda_dual=1, P^2 cancels, giving the witness: 2(PE - PD) + DD - EE.
    When lambda_dual=0, ensemble terms drop, giving standard MMD: P^2 - 2PD + DD.

    Args:
        params: IQP circuit parameters.
        iqp_circuit: IQP circuit.
        ground_truth: Training samples, rows of 0/1.
        visible_ops: Operators restricted to measured wires.
        all_ops: Full operator matrix.
        tr_ens: Weighted ensemble trace vector (stop_gradient).
        term_ens_ens: Unbiased E^2 self-term (stop_gradient scalar).
        n_samples: Circuit shots.
        key: JAX PRNG key.
        lambda_dual: Interpolation. 1=witness, 0=standalone MMD.
        init_coefs: Fixed initial gate coefficients.
        indep_estimates: Use independent sample estimates.
        return_components: Return component dict instead of scalar.
        max_batch_ops: Max operators per batch in op_expval.
    """
    tr_iqp_samples = iqp_circuit.op_expval(params, all_ops, n_samples, key, init_coefs, indep_estimates,
                                           return_samples=True, max_batch_ops=max_batch_ops,
                                           max_batch_samples=max_batch_samples)
    correction = jnp.mean(tr_iqp_samples**2, axis=-1) / n_samples
    tr_iqp = jnp.mean(tr_iqp_samples, axis=-1)
    tr_train = jnp.mean(1 - 2 * ((ground_truth @ visible_ops.T) % 2), axis=0)
    m = len(ground_truth)

    term_p_p = jnp.mean((tr_iqp * tr_iqp - correction) * n_samples / (n_samples - 1))
    cross_data = jnp.mean(tr_iqp * tr_train)
    cross_ens = jnp.mean(tr_iqp * tr_ens)
    term_data_data = jnp.mean((tr_train * tr_train * m - 1) / (m - 1))

    total = (1.0 - lambda_dual) * term_p_p + 2.0 * (lambda_dual * cross_ens - cross_data) + (term_data_data - lambda_dual * term_ens_ens)
    if return_components:
        return {
            'total': total,
            'data': -2.0 * cross_data + term_p_p + term_data_data,
            'ensemble': -2.0 * cross_ens + term_p_p + term_ens_ens,
        }
    return total

class EnsembleTerms:
    """Stores per-model traces and the shared operators they were computed on.
    
    Unlike mmd_loss_iqp (which samples ops fresh each call), the dual loss
    requires operators shared between ensemble traces and the new model.
    Operators are sampled once during add_term() and reused during training.
    """
    def __init__(self):
        self.trs = []    # trs[model_idx][sigma_idx] -> array (n_ops,)
        self.corrs = []  # corrs[model_idx][sigma_idx] -> array (n_ops,)
        self.ops = {}    # ops[sigma_idx] -> (all_ops, visible_ops)

    def sample_ops(self, iqp_circuit: IqpSimulator, sigma: float | list, n_ops: int,
                   key: Array, wires: list = None) -> None:
        """Sample and cache randomly drawn operators explicitly."""
        sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
        if wires is None:
            wires = list(range(iqp_circuit.n_qubits))

        self.ops.clear()
        for sigma_idx, s in enumerate(sigmas):
            key, subkey = jax.random.split(key, 2)
            self.ops[sigma_idx] = _make_ops(subkey, s, n_ops, iqp_circuit.n_qubits, wires)

    def add_term(self, params: jnp.ndarray, iqp_circuit: IqpSimulator,
                 sigma: float | list, n_ops: int, n_samples: int, key: Array,
                 init_coefs: list = None, wires: list = None, indep_estimates: bool = False,
                 max_batch_ops: int = None, max_batch_samples: int = None) -> None:
        """Evaluate a model on the cached operators and persist its traces."""
        init_coefs = jnp.array(init_coefs) if init_coefs is not None else None
        if wires is None:
            wires = list(range(iqp_circuit.n_qubits))

        sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
        tr_list, corr_list = [], []

        for sigma_idx, s in enumerate(sigmas):
            # Reuse ops if already sampled natively via sample_ops, else fallback sample
            if sigma_idx not in self.ops:
                key, subkey = jax.random.split(key, 2)
                self.ops[sigma_idx] = _make_ops(subkey, s, n_ops, iqp_circuit.n_qubits, wires)

            all_ops, _ = self.ops[sigma_idx]

            key, subkey = jax.random.split(key, 2)
            tr_iqp_samples = iqp_circuit.op_expval(
                params, all_ops, n_samples, subkey, init_coefs, indep_estimates,
                return_samples=True, max_batch_ops=max_batch_ops, max_batch_samples=max_batch_samples)
            tr_list.append(jnp.mean(tr_iqp_samples, axis=-1))
            corr_list.append(jnp.mean(tr_iqp_samples**2, axis=-1) / n_samples)

        self.trs.append(tuple(tr_list))
        self.corrs.append(tuple(corr_list))


def _weighted_ensemble(ensemble_terms: EnsembleTerms, weights: list, sigma_idx: int, n_samples: int,
                       stochastic_ops: bool = False, ensemble_models: list = None,
                       iqp_circuit: IqpSimulator = None, all_ops: jnp.ndarray = None, key: Array = None,
                       init_coefs: list = None, indep_estimates: bool = False,
                       max_batch_ops: int = None, max_batch_samples: int = None):
    """Compute weighted ensemble trace, correction vector, and unbiased self-term."""
    
    # 1. Stochastic mode: evaluate models dynamically on fresh ops
    if stochastic_ops and ensemble_models is not None:
        if not ensemble_models or not weights:
            n_ops = all_ops.shape[0] if all_ops is not None else 0
            zeros = jnp.zeros(n_ops)
            return zeros, zeros, jnp.float64(0.0)
            
        # Split key for each model to ensure independent samples across the ensemble
        subkeys = jax.random.split(key, len(ensemble_models))
        
        samples_enss = jnp.array([
            iqp_circuit.op_expval(m_params, all_ops, n_samples, skey, init_coefs, indep_estimates,
                                 return_samples=True, max_batch_ops=max_batch_ops,
                                 max_batch_samples=max_batch_samples)
            for m_params, skey in zip(ensemble_models, subkeys)
        ])
        tr_enss = jnp.mean(samples_enss, axis=-1)
        corr_enss = jnp.mean(samples_enss**2, axis=-1) / n_samples
        
        w = jnp.array(weights)
        if tr_enss.ndim == 1: tr_enss = tr_enss[None, :]
        if corr_enss.ndim == 1: corr_enss = corr_enss[None, :]
        if w.ndim == 0: w = w[None]
        
        tr_ens = jnp.sum(w[:, None] * tr_enss, axis=0)
        tr_ens_sq = jnp.einsum('i,ik,jk,j->k', w, tr_enss, tr_enss, w)
        corr = jnp.sum((w**2)[:, None] * corr_enss, axis=0)
        term_ens_ens = jnp.mean((tr_ens_sq - corr) * n_samples / (n_samples - 1))
        
        return (
            jax.lax.stop_gradient(tr_ens),
            jax.lax.stop_gradient(corr),
            jax.lax.stop_gradient(term_ens_ens),
        )

    # 2. Cached mode: use stored traces (must use stored ops!)
    if not ensemble_terms.trs or not weights:
        n_ops = ensemble_terms.ops[sigma_idx][0].shape[0] if (ensemble_terms and sigma_idx in ensemble_terms.ops) else 0
        zeros = jnp.zeros(n_ops)
        return zeros, zeros, jnp.float64(0.0)

    tr_enss = jnp.array([t[sigma_idx] for t in ensemble_terms.trs])
    corr_enss = jnp.array([c[sigma_idx] for c in ensemble_terms.corrs])
    w = jnp.array(weights)

    if tr_enss.ndim == 1: tr_enss = tr_enss[None, :]
    if corr_enss.ndim == 1: corr_enss = corr_enss[None, :]
    if w.ndim == 0: w = w[None]

    tr_ens = jnp.sum(w[:, None] * tr_enss, axis=0)
    tr_ens_sq = jnp.einsum('i,ik,jk,j->k', w, tr_enss, tr_enss, w)
    corr = jnp.sum((w**2)[:, None] * corr_enss, axis=0)
    term_ens_ens = jnp.mean((tr_ens_sq - corr) * n_samples / (n_samples - 1))

    return (
        jax.lax.stop_gradient(tr_ens),
        jax.lax.stop_gradient(corr),
        jax.lax.stop_gradient(term_ens_ens),
    )


def dual_mmd_loss(params: jnp.ndarray, iqp_circuit: IqpSimulator, ground_truth: jnp.ndarray,
                  ensemble_terms: EnsembleTerms, weights: list,
                  sigma: float | list, n_ops: int,
                  n_samples: int, key: Array, init_coefs: list = None, wires: list = None,
                  indep_estimates: bool = False, jit: bool = True,
                  lambda_dual: float = 1.0, return_components: bool = False,
                  return_traces: bool = False, stochastic_ops: bool = False,
                  ensemble_models: list = None,
                  max_batch_ops: int = None, max_batch_samples: int = None) -> float | dict:
    """Dual MMD boosting loss, averaged over Gaussian bandwidths.

    Like mmd_loss_iqp but with an ensemble repulsion term. Operators are
    shared with the ensemble traces stored in ensemble_terms.
    """
    if len(weights) == 0:
        lambda_dual = 0.0

    sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
    init_coefs = jnp.array(init_coefs) if init_coefs is not None else None

    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1")

    if wires is None:
        wires = list(range(iqp_circuit.n_qubits))

    losses = []
    trace_lists = {'trs_data': [], 'trs_ens': [], 'trs_iqp': [], 'trs_corr_ens': [], 'trs_corr_iqp': []}

    for sigma_idx, s in enumerate(sigmas):
        # 1. Operators
        if not stochastic_ops and ensemble_terms is not None and sigma_idx in ensemble_terms.ops:
            all_ops, visible_ops = ensemble_terms.ops[sigma_idx]
        else:
            key, subkey = jax.random.split(key, 2)
            all_ops, visible_ops = _make_ops(subkey, s, n_ops, iqp_circuit.n_qubits, wires)
            if not stochastic_ops and ensemble_terms is not None:
                ensemble_terms.ops[sigma_idx] = (all_ops, visible_ops)

        # 2. Weighted ensemble targets
        if len(weights) > 0:
            key, subkey = jax.random.split(key, 2)
            tr_ens, corr_ens, term_ens_ens = _weighted_ensemble(
                ensemble_terms, weights, sigma_idx, n_samples,
                stochastic_ops=stochastic_ops, ensemble_models=ensemble_models,
                iqp_circuit=iqp_circuit, all_ops=all_ops, key=subkey, 
                init_coefs=init_coefs, indep_estimates=indep_estimates,
                max_batch_ops=max_batch_ops, max_batch_samples=max_batch_samples
            )
        else:
            n_ops_cur = all_ops.shape[0]
            tr_ens = jnp.zeros(n_ops_cur)
            corr_ens = jnp.zeros(n_ops_cur)
            term_ens_ens = jnp.float64(0.0)

        if return_traces:
            key, subkey = jax.random.split(key, 2)
            tr_iqp_samples = iqp_circuit.op_expval(
                params, all_ops, n_samples, subkey, init_coefs, indep_estimates,
                return_samples=True, max_batch_ops=max_batch_ops, max_batch_samples=max_batch_samples)
            tr_iqp = jnp.mean(tr_iqp_samples, axis=-1)
            corr_iqp = jnp.mean(tr_iqp_samples**2, axis=-1) / n_samples
            tr_train = jnp.mean(1 - 2 * ((ground_truth @ visible_ops.T) % 2), axis=0)

            trace_lists['trs_data'].append(tr_train)
            trace_lists['trs_ens'].append(tr_ens)
            trace_lists['trs_iqp'].append(tr_iqp)
            trace_lists['trs_corr_ens'].append(corr_ens)
            trace_lists['trs_corr_iqp'].append(corr_iqp)
            continue

        # Core loss (JIT pattern matches mmd_loss_iqp)
        if iqp_circuit.sparse:
            loss_fn = dual_loss_estimate_iqp
        else:
            if jit:
                loss_fn = jax.jit(dual_loss_estimate_iqp, static_argnames=[
                    "iqp_circuit", "n_samples", "indep_estimates", "return_components",
                    "max_batch_ops", "max_batch_samples"])
            else:
                loss_fn = dual_loss_estimate_iqp

        losses.append(loss_fn(params, iqp_circuit, ground_truth, visible_ops, all_ops,
                              tr_ens, term_ens_ens, n_samples, key, lambda_dual,
                              init_coefs, indep_estimates, return_components,
                              max_batch_ops=max_batch_ops, max_batch_samples=max_batch_samples))

    if return_traces:
        return trace_lists

    if return_components:
        return {
            'total': sum(l['total'] for l in losses) / len(losses),
            'data': sum(l['data'] for l in losses) / len(losses),
            'ensemble': sum(l['ensemble'] for l in losses) / len(losses),
        }

    return sum(losses) / len(losses)


def gradient_snr(params: jnp.ndarray, iqp_circuit: IqpSimulator, x_train: jnp.ndarray,
                 loss_fn: callable, key: jax.Array, n_samples: int = 10, **loss_kwargs) -> dict:
    """Estimate Signal-to-Noise Ratio (SNR) of gradient estimates."""
    bound_loss_fn = functools.partial(
        loss_fn, iqp_circuit=iqp_circuit, x_train=x_train, **loss_kwargs)
    grad_fn = jax.grad(bound_loss_fn, argnums=0)
    gradient_norms = []

    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        grad_params = grad_fn(params, key=subkey)
        grad_norm = jnp.linalg.norm(grad_params)
        gradient_norms.append(float(grad_norm))

    gradient_norms = jnp.array(gradient_norms)
    mean_norm = float(jnp.mean(gradient_norms))
    std_norm = float(jnp.std(gradient_norms))
    snr = mean_norm / max(std_norm, 1e-10)

    return {
        'snr': snr,
        'mean_grad_norm': mean_norm,
        'std_grad_norm': std_norm,
        'gradient_norms': gradient_norms,
    }