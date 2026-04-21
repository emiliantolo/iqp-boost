"""Boosted ensemble of IQP circuits."""
import numpy as np
import jax
import jax.numpy as jnp
import iqpopt as iqp
from src.dual_mmd_loss import EnsembleTerms


class BoostedEnsemble:
    """Container for the boosted IQP ensemble state."""

    def __init__(self, iqp_circuit: iqp.IqpSimulator, n_models: int,
                 sigma: float | list, n_ops: int, n_samples: int,
                 lambda_dual: float = 1.0, wires: list = None,
                 max_batch_ops: int = None, max_batch_samples: int = None) -> None:
        self.iqp_circuit = iqp_circuit
        self.n_models = n_models
        self.sigma = sigma
        self.n_ops = n_ops
        self.n_samples = n_samples
        self.lambda_dual = lambda_dual
        self.wires = wires  # visible qubit indices (None = all qubits)
        self.max_batch_ops = max_batch_ops
        self.max_batch_samples = max_batch_samples
        self.terms = EnsembleTerms()
        self.weights: list[float] = []
        self.models: list[np.ndarray] = []
        self.training_losses: list = []

    def normalize_weights(self) -> None:
        if not self.weights:
            return
        weights = np.array(self.weights, dtype=float)
        total = weights.sum()
        if total <= 0:
            raise ValueError("ensemble weights must sum to a positive value")
        self.weights = (weights / total).tolist()

    def snapshot_state(self) -> dict:
        return {
            "weights": list(self.weights),
            "models": [np.array(m) for m in self.models],
            "terms_trs": [[np.array(x) for x in t_list] for t_list in self.terms.trs],
            "terms_corrs": [[np.array(x) for x in c_list] for c_list in self.terms.corrs],
            "terms_ops": {
                int(k): (np.array(v[0]), np.array(v[1]))
                for k, v in self.terms.ops.items()
            },
        }

    def restore_state(self, snapshot: dict) -> None:
        self.weights = list(snapshot["weights"])
        self.models = [np.array(m) for m in snapshot["models"]]
        self.terms.trs = [[jnp.array(y) for y in x] for x in snapshot["terms_trs"]]
        self.terms.corrs = [[jnp.array(y) for y in x] for x in snapshot["terms_corrs"]]
        self.terms.ops = {
            int(k): (jnp.array(v[0]), jnp.array(v[1]))
            for k, v in snapshot.get("terms_ops", {}).items()
        }

    def refresh_terms(self, key: jax.Array) -> None:
        """Clear cached operators/traces and re-evaluate all models on fresh operators.
        
        Used by 'step' caching level to ensure each new boosting step gets a completely 
        independent set of operators, while allowing fast fixed-operator cache lookups 
        during the inner training loop.
        """
        # Reset the underlying term traces
        self.terms.trs.clear()
        self.terms.corrs.clear()
        
        # Sample new explicit operators for this next step
        key, ops_key = jax.random.split(key)
        self.terms.sample_ops(self.iqp_circuit, self.sigma, self.n_ops, ops_key, wires=self.wires)
        
        # Re-evaluate all existing models (using the freshly sampled ops)
        for model_params in self.models:
            key, subkey = jax.random.split(key, 2)
            self.terms.add_term(
                model_params, self.iqp_circuit, self.sigma, self.n_ops,
                self.n_samples, subkey, wires=self.wires,
                max_batch_ops=self.max_batch_ops, max_batch_samples=self.max_batch_samples
            )

    def apply_weight_strategy(self, strategy: str,
                               trs_old: list[np.ndarray] = None,
                               trs_new: list[np.ndarray] = None,
                               trs_data: list[np.ndarray] = None,
                               trs_corr_old: list[np.ndarray] = None,
                               trs_corr_new: list[np.ndarray] = None,
                               samples_old: np.ndarray = None,
                               samples_new: np.ndarray = None,
                               ground_truth: np.ndarray = None,
                               validity_fn: callable = None,
                               coverage_fn: callable = None,
                               alpha_n_grid: int = 11,
                               validity_weight: float = 0.5) -> float:
        """Apply analytical weighting strategy. Returns the selected alpha."""
        if strategy in ('greedy', 'frank_wolfe'):
            alpha = self.weights[-1]
            print(f"  Weighting: {strategy} alpha = {alpha:.4f}")
            return float(alpha)

        from .utils import (
            compute_optimal_alpha_samples,
            compute_optimal_alpha_dual,
            compute_optimal_alpha_tvd_samples,
            compute_optimal_alpha_validity,
            compute_optimal_alpha_coverage,
            compute_optimal_alpha_validity_coverage_sum,
        )

        if strategy == 'line_search':
            if samples_old is not None and samples_new is not None and ground_truth is not None:
                alpha_opt = compute_optimal_alpha_samples(
                    samples_old, samples_new, ground_truth, self.sigma
                )
                print(f"  Weighting: sample line search alpha_opt = {alpha_opt:.4f}")
            elif trs_old is not None and trs_new is not None and trs_data is not None:
                alpha_opt = compute_optimal_alpha_dual(
                    trs_old, trs_new, trs_data,
                    trs_corr_old=trs_corr_old, trs_corr_new=trs_corr_new,
                    n_samples=self.n_samples
                )
                print(f"  Weighting: dual line search alpha_opt = {alpha_opt:.4f}")
            else:
                print("  [Warning] Missing data for line search. Falling back to greedy.")
                return float(self.weights[-1])

            if len(self.weights) >= 2:
                old_weight_sum = sum(self.weights[:-1])
                if old_weight_sum > 0:
                    scale = (1.0 - alpha_opt) / old_weight_sum
                    for i in range(len(self.weights) - 1):
                        self.weights[i] *= scale
                    self.weights[-1] = alpha_opt
                self.normalize_weights()
            return float(alpha_opt)

        if strategy == 'fully_corrective':
            from .utils import compute_optimal_weights_qp
            if trs_data is None:
                print("  [Warning] Missing data traces for QP. Falling back to greedy.")
                return float(self.weights[-1])

            w_opt = compute_optimal_weights_qp(
                all_trs=self.terms.trs,
                trs_data=trs_data,
                all_corrs=self.terms.corrs,
                n_samples=self.n_samples
            )

            self.weights = w_opt.tolist()
            w_str = ', '.join(f'{w:.4f}' for w in self.weights)
            print(f"  Weighting: fully corrective QP weights = [{w_str}]")
            return float(self.weights[-1])

        if strategy == 'tvd_line_search':
            if samples_old is None or samples_new is None or ground_truth is None:
                print("  [Warning] Missing samples/data for TVD line search. Falling back to greedy.")
                return float(self.weights[-1])

            alpha_opt = compute_optimal_alpha_tvd_samples(
                samples_old=samples_old,
                samples_new=samples_new,
                ground_truth=ground_truth,
            )
            print(f"  Weighting: TVD line search alpha_opt = {alpha_opt:.4f}")

            if len(self.weights) >= 2:
                old_weight_sum = sum(self.weights[:-1])
                if old_weight_sum > 0:
                    scale = (1.0 - alpha_opt) / old_weight_sum
                    for i in range(len(self.weights) - 1):
                        self.weights[i] *= scale
                    self.weights[-1] = alpha_opt
                self.normalize_weights()
            return float(alpha_opt)

        if strategy == 'validity_line_search':
            if samples_old is None or samples_new is None or validity_fn is None:
                print("  [Warning] Missing samples/validity_fn for validity line search. Falling back to greedy.")
                return float(self.weights[-1])

            alpha_opt = compute_optimal_alpha_validity(
                samples_old=samples_old,
                samples_new=samples_new,
                validity_fn=validity_fn,
                n_grid=int(alpha_n_grid),
            )
            print(f"  Weighting: validity line search alpha_opt = {alpha_opt:.4f}")

            if len(self.weights) >= 2:
                old_weight_sum = sum(self.weights[:-1])
                if old_weight_sum > 0:
                    scale = (1.0 - alpha_opt) / old_weight_sum
                    for i in range(len(self.weights) - 1):
                        self.weights[i] *= scale
                    self.weights[-1] = alpha_opt
                self.normalize_weights()
            return float(alpha_opt)

        if strategy == 'coverage_line_search':
            if samples_old is None or samples_new is None or ground_truth is None or coverage_fn is None:
                print("  [Warning] Missing samples/data/coverage_fn for coverage line search. Falling back to greedy.")
                return float(self.weights[-1])

            alpha_opt = compute_optimal_alpha_coverage(
                samples_old=samples_old,
                samples_new=samples_new,
                ground_truth=ground_truth,
                coverage_fn=coverage_fn,
                n_grid=int(alpha_n_grid),
            )
            print(f"  Weighting: coverage line search alpha_opt = {alpha_opt:.4f}")

            if len(self.weights) >= 2:
                old_weight_sum = sum(self.weights[:-1])
                if old_weight_sum > 0:
                    scale = (1.0 - alpha_opt) / old_weight_sum
                    for i in range(len(self.weights) - 1):
                        self.weights[i] *= scale
                    self.weights[-1] = alpha_opt
                self.normalize_weights()
            return float(alpha_opt)

        if strategy == 'sum_line_search':
            if (
                samples_old is None
                or samples_new is None
                or ground_truth is None
                or validity_fn is None
                or coverage_fn is None
            ):
                print("  [Warning] Missing inputs for validity/coverage sum line search. Falling back to greedy.")
                return float(self.weights[-1])

            alpha_opt = compute_optimal_alpha_validity_coverage_sum(
                samples_old=samples_old,
                samples_new=samples_new,
                ground_truth=ground_truth,
                validity_fn=validity_fn,
                coverage_fn=coverage_fn,
                validity_weight=float(validity_weight),
                n_grid=int(alpha_n_grid),
            )
            print(f"  Weighting: validity/coverage sum line search alpha_opt = {alpha_opt:.4f}")

            if len(self.weights) >= 2:
                old_weight_sum = sum(self.weights[:-1])
                if old_weight_sum > 0:
                    scale = (1.0 - alpha_opt) / old_weight_sum
                    for i in range(len(self.weights) - 1):
                        self.weights[i] *= scale
                    self.weights[-1] = alpha_opt
                self.normalize_weights()
            return float(alpha_opt)

        raise ValueError(f"Unknown weight_strategy: {strategy}")

    def add_model(self, params: np.ndarray, key: jax.Array, gamma: float = 2.0, tau: float = 2.0) -> float:
        """Add a trained model to the ensemble. Returns the initial alpha."""
        self.terms.add_term(
            params, self.iqp_circuit, self.sigma, self.n_ops,
            self.n_samples, key, wires=self.wires,
            max_batch_ops=self.max_batch_ops, max_batch_samples=self.max_batch_samples
        )

        it = len(self.models)
        if it == 0:
            alpha = 1.0
        else:
            alpha = min(1.0, gamma / (it + tau))
            
        self.weights = [(1.0 - alpha) * w for w in self.weights] + [alpha]
        self.models.append(params)
        self.normalize_weights()
        return float(alpha)

    def sample(self, n_samples: int, rng: np.random.Generator,
               return_details: bool = False,
               weights_override: np.ndarray | None = None) -> np.ndarray:
        """Sample from ensemble mixture."""
        weights_input = self.weights if weights_override is None else weights_override
        weights = np.array(weights_input, dtype=float)

        if weights.size == 0:
            raise ValueError("weights array is empty")
        weights = weights / weights.sum()

        if len(weights) != len(self.models):
            raise ValueError(f"Model count ({len(self.models)}) doesn't match weight count ({len(weights)})")

        model_indices = rng.choice(len(self.models), size=n_samples, p=weights)
        counts = np.bincount(model_indices, minlength=len(self.models))
        samples = []
        per_model_samples = [] if return_details else None
        for model_idx, count in enumerate(counts):
            if count > 0:
                model_samples = self.iqp_circuit.sample(self.models[model_idx], shots=int(count))
                samples.append(model_samples)
                if return_details:
                    per_model_samples.append(model_samples)
            elif return_details:
                per_model_samples.append(np.empty((0, self.iqp_circuit.n_qubits), dtype=int))

        samples = np.vstack(samples)

        # Slice out ancilla columns if wires are specified
        if self.wires is not None:
            samples = samples[:, self.wires]
            if return_details:
                per_model_samples = [s[:, self.wires] if len(s) > 0 else s for s in per_model_samples]

        if return_details:
            return samples, counts, per_model_samples
        return samples

    def save(self, path: str) -> None:
        import json
        data = {
            "weights": self.weights,
            "models": [m.tolist() for m in self.models],
            "sigma": self.sigma,
            "n_ops": self.n_ops,
            "lambda_dual": self.lambda_dual,
            "wires": self.wires,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
