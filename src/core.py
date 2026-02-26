"""Shared core functions for ensemble boosting (BAS and Gaussian datasets)."""

import numpy as np
import jax
import jax.numpy as jnp
import networkx as nx
import iqpopt as iqp
import iqpopt.gen_qml as gen
from iqpopt.gen_qml.sample_methods import mmd_loss_samples
from iqpopt.utils import nearest_neighbour_gates, local_gates, random_gates, initialize_from_data
from .ensemble import BoostedEnsemble
from .dual_mmd_loss import EnsembleTerms, mmd_loss_mt


def get_params_init(strategy: str, gates, data, key_rng=None):
    """
    Initialize parameters based on strategy.
    
    Args:
        strategy: 'covariance' or 'random'
        gates: Gate structure from IQP circuit
        data: Training data for covariance-based initialization
        key_rng: JAX random key for 'random' strategy
    
    Returns:
        Parameter array
    """
    if strategy == 'covariance':
        return initialize_from_data(gates, data)
    elif strategy == 'random':
        shape = (len(gates),)
        if key_rng is not None:
            return jax.random.uniform(key_rng, shape=shape, minval=-np.pi, maxval=np.pi)
        else:
            return np.random.uniform(-np.pi, np.pi, size=shape)
    else:
        raise ValueError(f"Unknown init_strategy: {strategy}")


def fast_binary_gaussian_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Fast Gaussian kernel for binary vectors using Hamming distance."""
    X_f = X.astype(np.float32)
    Y_f = Y.astype(np.float32)
    X_sum = np.sum(X_f, axis=1, keepdims=True)
    Y_sum = np.sum(Y_f, axis=1)
    H = X_sum + Y_sum - 2 * (X_f @ Y_f.T)
    return np.exp(-H / (2 * sigma**2))


def setup_iqp_circuit(n_qubits: int, topology: str = 'ring', **kwargs) -> tuple:
    """Configure IQP circuit gates based on topology.
    
    Args:
        n_qubits: Number of qubits
        topology: Gate structure ('neighbour', 'custom', etc.)
    
    Returns:
        (circuit, gates): IQP simulator and gate structure
    """
    if topology == 'neighbour':
        G = nx.cycle_graph(n_qubits)
        distance = kwargs.get('distance', 1)
        max_weight = kwargs.get('max_weight', 2)
        gates = nearest_neighbour_gates(G, distance=distance, max_weight=max_weight)
        desc = f"Qubits: {n_qubits}\nNeighbour topology: {len(gates)} parameters\n(distance={distance}, max_weight={max_weight})"
    elif topology == 'random':
        n_gates = kwargs.get('n_gates', n_qubits * 2)
        max_idx = kwargs.get('max_idx', n_qubits)
        min_weight = kwargs.get('min_weight', 1)
        max_weight = kwargs.get('max_weight', 2)
        gates = random_gates(n_gates, max_idx=max_idx, min_weight=min_weight, max_weight=max_weight)
        desc = f"Qubits: {n_qubits}\nRandom topology: {len(gates)} parameters\n(n_gates={n_gates},\
            max_idx={max_idx}, weight_range=[{min_weight},{max_weight}])"
    elif topology == 'local':
        max_weight = kwargs.get('max_weight', 2)
        gates = local_gates(n_qubits, max_weight=max_weight)
        desc = f"Qubits: {n_qubits}\nLocal topology: {len(gates)} parameters\n(max_weight={max_weight})"
    else:
        raise ValueError(f"Unknown topology: {topology}")

    circuit = iqp.IqpSimulator(n_qubits, gates)

    return circuit, gates, desc


def compute_kl_divergence(ground_truth: np.ndarray, model_samples: np.ndarray, 
                          n_bins: int = None, smoothing: float = 1e-10) -> float:
    """Compute KL divergence between discrete sample distributions.
    
    Args:
        ground_truth: Ground truth samples (binary vectors)
        model_samples: Model samples (binary vectors)
        n_bins: Number of bins (2^n_qubits). If None, inferred from data.
        smoothing: Laplace smoothing to avoid log(0)
    
    Returns:
        KL divergence D_KL(P_data || P_model)
    """
    ground_truth = np.asarray(ground_truth, dtype=int)
    model_samples = np.asarray(model_samples, dtype=int)
    
    # Convert binary vectors to integers
    def binary_to_int(samples):
        return np.sum(samples * (2 ** np.arange(samples.shape[1])), axis=1)
    
    gt_ints = binary_to_int(ground_truth)
    model_ints = binary_to_int(model_samples)
    
    if n_bins is None:
        n_bins = 2 ** ground_truth.shape[1]
    
    # Compute empirical distributions with Laplace smoothing
    gt_counts = np.bincount(gt_ints, minlength=n_bins) + smoothing
    model_counts = np.bincount(model_ints, minlength=n_bins) + smoothing
    
    p_data = gt_counts / gt_counts.sum()
    p_model = model_counts / model_counts.sum()
    
    # KL divergence
    kl = np.sum(p_data * np.log(p_data / p_model))
    return float(kl)


def compute_precision_recall_f1(ground_truth: np.ndarray, model_samples: np.ndarray,
                                sigma: float, threshold: float = None) -> dict:
    """Compute precision, recall, F1 using kernel-based matching.
    
    Args:
        ground_truth: Ground truth samples
        model_samples: Model samples
        sigma: Kernel bandwidth
        threshold: Match threshold (default: exp(-1) ~= 0.368)
    
    Returns:
        Dict with precision, recall, accuracy, f_score
    """
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
    accuracy = (np.mean(max_kernel_per_data) + np.mean(max_kernel_per_model)) / 2
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'f_score': float(f_score),
    }


def compute_metrics(ground_truth: np.ndarray, model_samples: np.ndarray, 
                    validity_fn: callable, coverage_fn: callable) -> dict:
    """Compute validity and coverage metrics using provided validator functions.
    
    Args:
        ground_truth: Ground truth samples
        model_samples: Model samples
        validity_fn: Function that takes model_samples and returns validity rate
        coverage_fn: Function that takes (ground_truth, model_samples) and returns coverage
    
    Returns:
        Dict with validity_rate and coverage
    """
    ground_truth = np.asarray(ground_truth)
    model_samples = np.asarray(model_samples)
    
    validity_rate = validity_fn(model_samples)
    coverage = coverage_fn(ground_truth, model_samples)
    return {'validity_rate': validity_rate, 'coverage': coverage}


def sample_ensemble(ensemble: BoostedEnsemble, circuit: iqp.IqpSimulator, n_samples: int,
                    rng: np.random.Generator, return_details: bool = False,
                    weights_override: np.ndarray | None = None):
    """Sample from ensemble mixture with optional per-model breakdown."""
    weights_input = ensemble.weights if weights_override is None else weights_override
    weights = np.array(weights_input, dtype=float)
    
    if weights.size == 0:
        raise ValueError("weights array is empty")
    
    weights = weights / weights.sum()
    
    if len(weights) != len(ensemble.models):
        raise ValueError(f"Model count ({len(ensemble.models)}) doesn't match weight count ({len(weights)})")
    
    model_indices = rng.choice(len(ensemble.models), size=n_samples, p=weights)
    counts = np.bincount(model_indices, minlength=len(ensemble.models))
    samples = []
    per_model_samples = [] if return_details else None
    for model_idx, count in enumerate(counts):
        if count > 0:
            model_samples = circuit.sample(ensemble.models[model_idx], shots=int(count))
            samples.append(model_samples)
            if return_details:
                per_model_samples.append(model_samples)
        elif return_details:
            per_model_samples.append(np.empty((0, circuit.n_qubits), dtype=int))
    samples = np.vstack(samples)
    if return_details:
        return samples, counts, per_model_samples
    return samples


def evaluate_samples(ground_truth: np.ndarray, samples: np.ndarray, sigma: float,
                     validity_fn: callable, coverage_fn: callable,
                     compute_kl: bool = True) -> dict:
    """Evaluate all metrics: MMD, KL, validity, coverage, precision, recall, F1.
    
    Args:
        ground_truth: Ground truth samples
        samples: Model samples
        sigma: Kernel bandwidth
        validity_fn: Function that takes model_samples and returns validity rate
        coverage_fn: Function that takes (ground_truth, model_samples) and returns coverage
        compute_kl: Whether to compute KL divergence
    
    Returns:
        Dict with all metrics
    """
    mmd = float(mmd_loss_samples(ground_truth, samples, sigma))
    
    # Validity and coverage
    metrics = compute_metrics(ground_truth, samples, validity_fn, coverage_fn)
    
    # Precision, recall, F1
    prf_metrics = compute_precision_recall_f1(ground_truth, samples, sigma)
    
    # KL divergence
    kl = compute_kl_divergence(ground_truth, samples) if compute_kl else 0.0
    
    return {
        'mmd': mmd,
        'kl': kl,
        'validity': metrics['validity_rate'],
        'coverage': metrics['coverage'],
        'precision': prf_metrics['precision'],
        'recall': prf_metrics['recall'],
        'accuracy': prf_metrics['accuracy'],
        'f_score': prf_metrics['f_score'],
    }


def compute_optimal_step_size(old_ensemble_samples: np.ndarray, new_model_samples: np.ndarray, 
                               ground_truth: np.ndarray, sigma: float, 
                               term_data_data: float = None,
                               alpha_min: float = 0.0, alpha_max: float = 1.0) -> tuple[float, dict]:
    """
    Compute optimal mixture step size by analytically minimizing MMD^2(alpha).
    
    For mixture (1-alpha)*F_old + alpha*f_new, MMD^2(alpha) is quadratic in alpha.
    Finds minimum analytically: alpha_opt = -b/(2a), clipped to [alpha_min, alpha_max].
    """
    K_old_old = fast_binary_gaussian_kernel(old_ensemble_samples, old_ensemble_samples, sigma)
    K_new_new = fast_binary_gaussian_kernel(new_model_samples, new_model_samples, sigma)
    K_old_new = fast_binary_gaussian_kernel(old_ensemble_samples, new_model_samples, sigma)
    K_data_old = fast_binary_gaussian_kernel(ground_truth, old_ensemble_samples, sigma)
    K_data_new = fast_binary_gaussian_kernel(ground_truth, new_model_samples, sigma)
    
    term_old_old = float(np.mean(K_old_old))
    term_new_new = float(np.mean(K_new_new))
    term_old_new = float(np.mean(K_old_new))
    term_data_old = float(np.mean(K_data_old))
    term_data_new = float(np.mean(K_data_new))
    
    if term_data_data is None:
        K_data_data = fast_binary_gaussian_kernel(ground_truth, ground_truth, sigma)
        term_data_data = float(np.mean(K_data_data))
    
    # Coefficients of MMD^2(alpha) = a*alpha^2 + b*alpha + c
    a = term_old_old - 2*term_old_new + term_new_new 
    b = -2*term_old_old + 2*term_old_new - 2*term_data_new + 2*term_data_old
    c = term_old_old - 2*term_data_old + term_data_data
    
    if abs(a) < 1e-12:
        alpha_opt = alpha_max if b < 0 else alpha_min
    else:
        alpha_opt = -b / (2 * a)
        alpha_opt = np.clip(alpha_opt, alpha_min, alpha_max)
    
    mmd_sq_at_opt = a * alpha_opt**2 + b * alpha_opt + c
    mmd_sq_at_0 = c
    mmd_sq_at_1 = a + b + c
    
    info = {
        'alpha_opt': alpha_opt,
        'improvement_vs_0': mmd_sq_at_0 - mmd_sq_at_opt,
    }
    
    return float(alpha_opt), info


def current_terms_snapshot(ensemble: BoostedEnsemble) -> dict | None:
    """Snapshot current ensemble terms for loss evaluation."""
    if not ensemble.terms.trs:
        return None
    return {
        "trs": np.array(ensemble.terms.trs),
        "corrs": np.array(ensemble.terms.corrs),
        "weights": list(ensemble.weights),
    }


def evaluate_terms_against_snapshot(model_params: np.ndarray, ensemble: BoostedEnsemble,
                                    ground_truth: np.ndarray, key: jax.Array,
                                    terms_snapshot: dict | None) -> tuple[float, float, float]:
    """Evaluate data_loss, ensemble_loss, and dual loss against snapshot."""
    key, subkey_data, subkey_ens = jax.random.split(key, 3)
    data_loss = gen.mmd_loss_iqp(
        model_params, ensemble.iqp_circuit, ground_truth, ensemble.sigma,
        ensemble.n_ops, ensemble.n_samples, subkey_data,
    )
    if terms_snapshot is None or len(terms_snapshot.get("trs", [])) == 0:
        return float(data_loss), float("nan"), float("nan")
    terms = EnsembleTerms()
    terms.trs = np.array(terms_snapshot["trs"])
    terms.corrs = np.array(terms_snapshot["corrs"])
    ens_loss = mmd_loss_mt(
        model_params, ensemble.iqp_circuit, terms, terms_snapshot["weights"],
        ensemble.sigma, ensemble.n_ops, ensemble.n_samples, subkey_ens,
    )
    dual = data_loss - ensemble.lambda_dual * ens_loss
    return float(data_loss), float(ens_loss), float(dual)


def ensemble_loss_against_snapshot(model_params: np.ndarray, ensemble: BoostedEnsemble,
                                   key: jax.Array, terms_snapshot: dict | None,
                                   weights_override: np.ndarray | None = None) -> float:
    """Compute ensemble loss against snapshot with optional weight override."""
    if terms_snapshot is None or len(terms_snapshot.get("trs", [])) == 0:
        return float("nan")
    terms = EnsembleTerms()
    terms.trs = np.array(terms_snapshot["trs"])
    terms.corrs = np.array(terms_snapshot["corrs"])
    if weights_override is None:
        weights = np.array(terms_snapshot["weights"], dtype=float)
    else:
        weights = np.array(weights_override, dtype=float)
    weights = weights / weights.sum()
    ens_loss = mmd_loss_mt(
        model_params, ensemble.iqp_circuit, terms, weights,
        ensemble.sigma, ensemble.n_ops, ensemble.n_samples, key,
    )
    return float(ens_loss)


def data_loss_series(params_hist: list, iqp_circuit: iqp.IqpSimulator, ground_truth: np.ndarray,
                     sigma: float, n_ops: int, n_samples: int, key: jax.Array) -> list:
    """Compute data loss trajectory over parameter history."""
    if not params_hist:
        return []
    losses = []
    key_iter = key
    for params in params_hist:
        key_iter, subkey = jax.random.split(key_iter, 2)
        loss = gen.mmd_loss_iqp(
            params, iqp_circuit, ground_truth, sigma, n_ops, n_samples, subkey,
        )
        losses.append(float(loss))
    return losses


def ensemble_loss_series(params_hist: list, terms_snapshot: dict | None, iqp_circuit: iqp.IqpSimulator,
                         sigma: float, n_ops: int, n_samples: int, key: jax.Array) -> list:
    """Compute ensemble loss trajectory over parameter history."""
    if not params_hist or terms_snapshot is None:
        return []
    terms = EnsembleTerms()
    terms.trs = np.array(terms_snapshot["trs"])
    terms.corrs = np.array(terms_snapshot["corrs"])
    weights = terms_snapshot["weights"]
    losses = []
    key_iter = key
    for params in params_hist:
        key_iter, subkey = jax.random.split(key_iter, 2)
        loss = mmd_loss_mt(
            params, iqp_circuit, terms, weights, sigma, n_ops, n_samples, subkey,
        )
        losses.append(float(loss))
    return losses


def apply_weight_strategy(ensemble: BoostedEnsemble, snapshot: dict, rng: np.random.Generator,
                          circuit: iqp.IqpSimulator, ground_truth: np.ndarray, 
                          sigma: float, eval_samples: int,
                          strategy: str = 'greedy') -> None:
    """
    Apply weight strategy (greedy or line_search) to ensemble.
    
    Updates ensemble.weights in-place.
    """
    if strategy == 'greedy':
        # Already set by ensemble.step(), just report
        fw_alpha = ensemble.weights[-1]
        print(f"  Weighting: standard greedy alpha = {fw_alpha:.4f} = 2/(2+{len(ensemble.weights)-1})")
        print(f"    Weights: {[f'{w:.3f}' for w in ensemble.weights]}")
    
    elif strategy == 'line_search':
        print(f"  Weighting: line search (analytical quadratic optimization)...")
        # Sample from old ensemble (BEFORE adding new model)
        old_models = snapshot['models']
        old_weights = np.array(snapshot['weights'])
        old_weights = old_weights / old_weights.sum()
        
        # Sample from old ensemble
        old_model_counts = rng.choice(len(old_models), size=eval_samples, p=old_weights)
        old_ens_samples = []
        for model_idx, count in enumerate(np.bincount(old_model_counts, minlength=len(old_models))):
            if count > 0:
                model_samples = circuit.sample(old_models[model_idx], shots=int(count))
                old_ens_samples.append(model_samples)
        old_ens_samples = np.vstack(old_ens_samples) if old_ens_samples else np.empty((0, circuit.n_qubits), dtype=int)
        
        # Sample from new model only
        new_model_samples = circuit.sample(ensemble.models[-1], shots=eval_samples)
        
        # Compute optimal alpha
        alpha_opt, opt_info = compute_optimal_step_size(
            old_ens_samples, new_model_samples, ground_truth, sigma,
            alpha_min=0.0, alpha_max=1.0
        )
        
        improvement = opt_info['improvement_vs_0']
        print(f"    Optimal alpha = {alpha_opt:.4f} (MMD improvement: {improvement:.6f})")
        
        # Apply optimal step size
        trial_weights = ensemble.weights.copy()
        if len(trial_weights) >= 2:
            old_weight_sum = sum(trial_weights[:-1])
            if old_weight_sum > 0:
                scale = (1.0 - alpha_opt) / old_weight_sum
                trial_weights[:-1] = [w * scale for w in trial_weights[:-1]]
                trial_weights[-1] = alpha_opt
                total = sum(trial_weights)
                trial_weights = [w / total for w in trial_weights]
        
        ensemble.weights = trial_weights
        ensemble.normalize_weights()
        print(f"    Weights: {[f'{w:.3f}' for w in ensemble.weights]}")
    
    else:
        raise ValueError(f"Unknown weight_strategy: {strategy}. Must be one of: 'greedy', 'line_search'")


def check_acceptance(current_mmd: float, previous_mmd: float, ensemble: BoostedEnsemble,
                     snapshot: dict, step: int, keep_all: bool = False, 
                     stop_on_reject: bool = False) -> tuple[bool, bool]:
    """Check if new model should be accepted based on MMD improvement.
    
    Args:
        current_mmd: Current ensemble MMD
        previous_mmd: Previous ensemble MMD
        ensemble: Ensemble object (for potential restore)
        snapshot: Saved state snapshot (for potential restore)
        step: Current step number
        keep_all: If True, keep all models (diagnostic mode)
        stop_on_reject: If True, stop boosting on first rejection
    
    Returns:
        (accepted, should_stop): Whether to accept the model and whether to stop boosting
    """
    delta_mmd = previous_mmd - current_mmd
    
    if keep_all:
        # Diagnostic mode: keep all models
        if delta_mmd <= 0:
            print(f"  [WARNING] MMD worsened but kept (diagnostic, delta={delta_mmd:+.6f})")
        else:
            print(f"  [OK] MMD improved (delta={delta_mmd:+.6f})")
        return True, False
    
    # Normal mode: require improvement
    if delta_mmd <= 0:
        print(f"  [REJECT] MMD did not improve (delta={delta_mmd:+.6f})")
        ensemble.restore_state(snapshot)
        if stop_on_reject:
            print(f"  Stopping boosting (stop_on_reject=True)")
            return False, True
        return False, False
    
    print(f"  [ACCEPT] MMD improved (delta={delta_mmd:+.6f})")
    return True, False

