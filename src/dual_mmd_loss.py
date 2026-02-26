import jax
import jax.numpy as jnp
from iqpopt import IqpSimulator
from iqpopt.gen_qml import mmd_loss_iqp
from jax._src.typing import Array
import functools
import numpy as np

def log_operator_term_sizes(all_ops: jnp.ndarray, sigma: float | list, label: str = "Operators") -> None:
    """Log the average term size (number of measured qubits) for operators.
    
    Args:
        all_ops: Operator matrix of shape (n_ops, n_qubits) with binary entries
        sigma: Kernel width parameter (for context logging)
        label: Label for the log output
    """
    # Convert to numpy if needed
    ops_np = np.array(all_ops)
    # Term size = number of 1s in each operator
    term_sizes = ops_np.sum(axis=1)
    avg_term_size = float(np.mean(term_sizes))
    max_term_size = int(np.max(term_sizes))
    min_term_size = int(np.min(term_sizes))
    n_ops = ops_np.shape[0]
    n_qubits = ops_np.shape[1]
    
    print(f"[{label}] n_ops={n_ops}, n_qubits={n_qubits}, avg_term_size={avg_term_size:.2f}, std_term_size={np.std(term_sizes):.2f}, " \
          f"min={min_term_size}, max={max_term_size}, sigma={sigma}")

def get_ops_mmd_loss(iqp_circuit: IqpSimulator, sigma: float | list, n_ops: int,
                     key: jax.Array, wires: list = None, verbose: bool = False) -> jnp.ndarray:
    """Generate random operators for MMD loss computation."""
    if wires is None:
        wires = list(range(iqp_circuit.n_qubits))
    
    p_MMD = (1-jnp.exp(-1/2/sigma**2))/2
    visible_ops = jnp.array(jax.random.binomial(
        key, 1, p_MMD, shape=(n_ops, len(wires))), dtype='float64')

    all_ops = []
    i = 0
    for q in range(iqp_circuit.n_qubits):
        if q in wires:
            all_ops.append(visible_ops[:, i])
            i += 1
        else:
            all_ops.append(jnp.zeros(n_ops))
    all_ops = jnp.array(all_ops, dtype='float64').T

    if verbose:
        log_operator_term_sizes(all_ops, sigma, "Generated operators")

    return all_ops

def get_tr_iqp(params: jnp.ndarray, iqp_circuit: IqpSimulator,
                      all_ops: jnp.ndarray, n_samples: int, key: Array, init_coefs: list = None, indep_estimates: bool = False) -> float:
    """Compute trace and correction term for MMD loss."""
    tr_iqp_samples = iqp_circuit.op_expval(params, all_ops, n_samples, key, init_coefs, indep_estimates, return_samples=True)
    correction = jnp.mean(tr_iqp_samples**2, axis=-1)/n_samples
    tr_iqp = jnp.mean(tr_iqp_samples, axis=-1)
    return tr_iqp, correction

class EnsembleTerms():

    def __init__(self):
        self.trs = []
        self.corrs = []
        self.cached_ops: dict[tuple, jnp.ndarray] = {}

    def clear_operator_cache(self) -> None:
        """Clear cached operators."""
        self.cached_ops = {}

    def add_term(self, params: jnp.ndarray, iqp_circuit: IqpSimulator, sigma: float | list, n_ops: int,
                    n_samples: int, key: jax.Array, init_coefs: list = None, wires: list = None,
                    indep_estimates: bool = False, jit: bool = True, verbose: bool = False) -> float:
        """Add model term to ensemble."""
        sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
        init_coefs = jnp.array(init_coefs) if init_coefs is not None else None

        if n_samples <= 1:
            raise ValueError("n_samples must be greater than 1")

        if wires is None:
            wires = list(range(iqp_circuit.n_qubits))

        terms = []
        for sigma in sigmas:
            key, op_key, sample_key = jax.random.split(key, 3)
            cache_key = (float(sigma), int(n_ops), tuple(wires))
            if cache_key not in self.cached_ops:
                self.cached_ops[cache_key] = get_ops_mmd_loss(iqp_circuit, sigma, n_ops, op_key, wires, verbose=verbose)
            all_ops = self.cached_ops[cache_key]

            if iqp_circuit.sparse:
                exp = get_tr_iqp
            else:
                if jit:
                    exp = jax.jit(get_tr_iqp, static_argnames=[
                                "iqp_circuit", "n_samples", "indep_estimates"])
                else:
                    exp = get_tr_iqp

            terms.append(exp(params, iqp_circuit, all_ops, n_samples, sample_key, init_coefs, indep_estimates))

        tr_enss, corr_enss = zip(*terms)
        tr_enss, corr_enss = jnp.array(tr_enss), jnp.array(corr_enss)
        self.trs.append(jnp.mean(tr_enss, axis=0))
        self.corrs.append(jnp.mean(corr_enss, axis=0))

def loss_estimate_mt(params: jnp.ndarray, iqp_circuit: IqpSimulator, ensemble_terms: EnsembleTerms, weights: list,
                      all_ops: jnp.ndarray, n_samples: int, key: Array, init_coefs: list = None, indep_estimates: bool = False, sqrt_loss: bool = False,
                      return_expvals: bool = False) -> float:
    """Estimate MMD loss against ensemble."""
    tr_iqp, correction = get_tr_iqp(params, iqp_circuit, all_ops, n_samples, key, init_coefs, indep_estimates)

    tr_enss, corr_enss = ensemble_terms.trs, ensemble_terms.corrs
    tr_enss, corr_enss = jnp.array(tr_enss), jnp.array(corr_enss)
    weights = jnp.array(weights)

    # Ensure 2D shape (n_models, n_ops) even for a single model
    if tr_enss.ndim == 1:
        tr_enss = tr_enss[None, :]
    if corr_enss.ndim == 1:
        corr_enss = corr_enss[None, :]
    if weights.ndim == 0:
        weights = weights[None]
    if tr_enss.shape[0] != weights.shape[0]:
        raise ValueError(
            f"ensemble terms/models mismatch: tr_enss has {tr_enss.shape[0]} models, weights has {weights.shape[0]}"
        )

    # Weighted average of model traces
    tr_ens = jnp.average(tr_enss, axis=0, weights=weights)
    
    # Weighted ensemble self-term
    tr_ens_sq = jnp.einsum('i,ik,jk,j->k', weights, tr_enss, tr_enss, weights)
    
    # Weighted correction term
    corr_ens_sq = jnp.sum((weights**2)[:, None] * corr_enss, axis=0)

    # Unbiased MMD estimate
    res = (tr_iqp*tr_iqp-correction)*n_samples/(n_samples-1) - 2*tr_iqp*tr_ens + (tr_ens_sq-corr_ens_sq)*n_samples/(n_samples-1)

    res = jnp.mean(res) if not return_expvals else res
    res = jnp.sqrt(jnp.abs(res)) if sqrt_loss else res

    return res

def mmd_loss_mt(params: jnp.ndarray, iqp_circuit: IqpSimulator, ensemble_terms: EnsembleTerms, weights: list, sigma: float | list, n_ops: int,
                 n_samples: int, key: Array, init_coefs: list = None, wires: list = None, indep_estimates: bool = False, jit: bool = True,
                 sqrt_loss: bool = False, return_expvals: bool = False, verbose: bool = False) -> float:
    """Compute MMD loss of model against ensemble."""
    sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
    init_coefs = jnp.array(init_coefs) if init_coefs is not None else None

    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1")

    if wires is None:
        wires = list(range(iqp_circuit.n_qubits))

    losses = []
    for sigma in sigmas:
        key, op_key, sample_key = jax.random.split(key, 3)
        cache_key = (float(sigma), int(n_ops), tuple(wires))
        if cache_key not in ensemble_terms.cached_ops:
            ensemble_terms.cached_ops[cache_key] = get_ops_mmd_loss(iqp_circuit, sigma, n_ops, op_key, wires, verbose=verbose)
        all_ops = ensemble_terms.cached_ops[cache_key]

        if iqp_circuit.sparse:
            loss = loss_estimate_mt
        else:
            if jit:
                loss = jax.jit(loss_estimate_mt, static_argnames=[
                               "iqp_circuit", "ensemble_terms", "n_samples", "indep_estimates", "sqrt_loss", "return_expvals"])
            else:
                loss = loss_estimate_mt

        losses.append(loss(params, iqp_circuit, ensemble_terms, weights, all_ops, n_samples, sample_key, init_coefs, indep_estimates,
                          sqrt_loss, return_expvals=return_expvals))

    if return_expvals:
        return losses
    else:
        return sum(losses)/len(losses)

def dual_mmd_loss(params: jnp.ndarray, iqp_circuit: IqpSimulator, ground_truth: jnp.ndarray, ensemble_terms: EnsembleTerms, weights: list, sigma: float | list, n_ops: int,
                 n_samples: int, key: Array, init_coefs: list = None, wires: list = None, indep_estimates: bool = False, jit: bool = True,
                 sqrt_loss: bool = False, return_expvals: bool = False,
                 lambda_dual: float = 1.0, return_components: bool = False, verbose: bool = False) -> float | dict:
    """Functional gradient boosting loss: data_loss - lambda * ensemble_loss.
    
    Trains new models to maximize distance from current ensemble (repulsion).
    
    Args:
        return_components: If True, returns dict with 'total', 'data', 'ensemble' keys
    """
    
    key, subkey = jax.random.split(key, 2)
    data_loss = mmd_loss_iqp(params, iqp_circuit, ground_truth, sigma, n_ops, n_samples, subkey, init_coefs, wires, indep_estimates)
    ens_loss = mmd_loss_mt(params, iqp_circuit, ensemble_terms, weights, sigma, n_ops, n_samples, subkey, init_coefs, wires, indep_estimates, verbose=verbose)
    
    if return_components:
        return {
            'total': data_loss - lambda_dual * ens_loss,
            'data': data_loss,
            'ensemble': ens_loss
        }
    return data_loss - lambda_dual * ens_loss


def gradient_snr(params: jnp.ndarray, iqp_circuit: IqpSimulator, x_train: jnp.ndarray,
                 loss_fn: callable, key: jax.Array, n_samples: int = 10, **loss_kwargs) -> dict:
    """
    Estimate Signal-to-Noise Ratio (SNR) of gradient estimates.
    
    Computes gradients with different random seeds to measure consistency.
    High SNR indicates stable, reliable gradients.
    
    Args:
        params: Current model parameters
        iqp_circuit: The IqpSimulator circuit
        x_train: Training data
        loss_fn: Loss function taking (params, iqp_circuit, x_train, key, ...) returns scalar
        key: JAX random key
        n_samples: Number of gradient samples for SNR estimate (default 10)
        **loss_kwargs: Additional arguments to loss_fn
    
    Returns:
        dict with: 'snr' (mean/std ratio), 'mean_grad_norm', 'std_grad_norm', 'gradient_norms'
    """
    bound_loss_fn = functools.partial(
        loss_fn,
        iqp_circuit=iqp_circuit,
        x_train=x_train,
        **loss_kwargs
    )
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
