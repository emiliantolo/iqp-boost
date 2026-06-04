"""Boosted ensemble of IQP circuits."""
import numpy as np
import jax
import jax.numpy as jnp
import iqpopt as iqp
from src.core.dual_mmd_loss import EnsembleTerms


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

    @classmethod
    def load(cls, path: str, iqp_circuit, n_samples: int, max_batch_ops: int = None, max_batch_samples: int = None):
        """Reconstruct a BoostedEnsemble from a JSON saved by ``save()``.

        Args:
            path: Path to the JSON file.
            iqp_circuit: IqpSimulator instance (must match the original circuit).
            n_samples: Number of circuit shots to use for trace estimates.
            max_batch_ops: Optional batching limit for operators.
            max_batch_samples: Optional batching limit for samples.
        """
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        sigma = data["sigma"]
        n_ops = int(data["n_ops"])
        lambda_dual = float(data["lambda_dual"])
        wires = data["wires"]
        instance = cls(
            iqp_circuit=iqp_circuit,
            n_models=len(data["models"]),
            sigma=sigma,
            n_ops=n_ops,
            n_samples=n_samples,
            lambda_dual=lambda_dual,
            wires=wires,
            max_batch_ops=max_batch_ops,
            max_batch_samples=max_batch_samples,
        )
        instance.weights = [float(w) for w in data["weights"]]
        instance.models = [np.array(m) for m in data["models"]]
        return instance
