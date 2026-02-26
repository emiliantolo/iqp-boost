import numpy as np
import jax
import jax.numpy as jnp
import iqpopt as iqp
import iqpopt.gen_qml as gen
from .dual_mmd_loss import EnsembleTerms, dual_mmd_loss


class BoostedEnsemble:
    def __init__(self, iqp_circuit: iqp.IqpSimulator, n_models: int, sigma: float | list, n_ops: int, n_samples: int,
        init_coefs: list | None = None, wires: list | None = None, indep_estimates: bool = False,
        cache_ensemble_traces: bool = False, ensemble_samples_coeff: float = 10.0, lambda_dual: float = 1.0) -> None:
        self.iqp_circuit = iqp_circuit
        self.n_models = n_models
        self.sigma = sigma
        self.n_ops = n_ops
        self.n_samples = n_samples
        self.init_coefs = init_coefs
        self.wires = wires
        self.indep_estimates = indep_estimates
        self.cache_ensemble_traces = cache_ensemble_traces
        self.ensemble_samples_coeff = ensemble_samples_coeff
        self.lambda_dual = lambda_dual
        self.terms = EnsembleTerms()
        self.weights: list[float] = []
        self.models: list[np.ndarray] = []
        self.training_losses: list = []  # Store training losses for each model
        self._model_count = 0  # Track number of models trained

    def normalize_weights(self) -> None:
        weights = np.array(self.weights, dtype=float)
        total = weights.sum()
        if total <= 0:
            raise ValueError("ensemble weights must sum to a positive value")
        self.weights = (weights / total).tolist()
        self.terms.weights = self.weights

    def snapshot_state(self) -> dict:
        return {
            "weights": list(self.weights),  # Create fresh list copy
            "models": [np.array(m) for m in self.models],  # Deep copy models
            "terms_trs": [np.array(t) for t in self.terms.trs],
            "terms_corrs": [np.array(c) for c in self.terms.corrs],
        }

    def restore_state(self, snapshot: dict) -> None:
        self.weights = list(snapshot["weights"])  # Always create fresh list copy on restore
        self.models = [np.array(m) for m in snapshot["models"]]  # Fresh copy of models
        self.terms.trs = [np.array(t) for t in snapshot["terms_trs"]]
        self.terms.corrs = [np.array(c) for c in snapshot["terms_corrs"]]
        self.terms.weights = self.weights  # Sync with new weights list

    def add_model(self, params: np.ndarray, key: jax.Array) -> None:
        if self.cache_ensemble_traces:
            term_samples = int(self.n_samples * self.ensemble_samples_coeff)
        else:
            term_samples = self.n_samples
        term_samples = max(2, term_samples)
        self.terms.add_term(params, self.iqp_circuit, self.sigma, self.n_ops, term_samples,
                            key, init_coefs=self.init_coefs, wires=self.wires, indep_estimates=self.indep_estimates)
        before_add = len(self.weights)
        self.update_weights(len(self.weights))
        self.models.append(params)
        self._model_count += 1  # Increment model counter
        
        # Sanity check
        if len(self.weights) != len(self.models):
            raise RuntimeError(
                f"CRITICAL: After add_model, weights and models are out of sync! "
                f"weights={len(self.weights)} (was {before_add}), "
                f"models={len(self.models)}, "
                f"terms has {len(self.terms.trs)} terms"
            )

    def update_weights(self, it) -> None:
        # Heuristic Frank-Wolfe
        alpha = 2 / (it + 2)
        self.weights = [(1 - alpha) * w for w in self.weights] + [alpha]
        self.normalize_weights()

    def train_base(self, ground_truth: np.ndarray, key: jax.Array, steps: int = 20, stepsize: float = 0.05,
                   monitor_interval: int | None = None, init_strategy: str = 'covariance', turbo: int = None) -> jax.Array:
        from .core import get_params_init
        params_init = get_params_init(init_strategy, self.iqp_circuit.gates, ground_truth, key)
        loss_kwargs = {
            "params": params_init,
            "iqp_circuit": self.iqp_circuit,
            "ground_truth": ground_truth,
            "sigma": self.sigma,
            "n_ops": self.n_ops,
            "n_samples": self.n_samples,
        }
        trainer = iqp.Trainer("Adam", gen.mmd_loss_iqp, stepsize=stepsize)
        trainer.train(n_iters=steps, loss_kwargs=loss_kwargs, monitor_interval=monitor_interval, turbo=turbo)
        key, subkey = jax.random.split(key, 2)
        self.training_losses.append({
            'total': np.array(trainer.losses), 
            'data_final': None,  # baseline is pure data loss
            'ensemble_final': None
        })
        self.add_model(trainer.final_params, subkey)
        return key

    def step(self, ground_truth: np.ndarray, key: jax.Array, steps: int = 20, stepsize: float = 0.05,
             verbose: bool = False, monitor_interval: int | None = None, init_strategy: str = 'random') -> jax.Array:
        from .core import get_params_init
        key, step_key = jax.random.split(key)
        step_key, init_key = jax.random.split(step_key)
        
        params_init = get_params_init(init_strategy, self.iqp_circuit.gates, ground_truth, init_key)
        
        # If not caching, compute fresh ensemble terms for this step
        ensemble_terms = self.terms
        if not self.cache_ensemble_traces and len(self.models) > 0:
            ensemble_terms = EnsembleTerms()
            step_key_inner = step_key
            for model_params in self.models:
                term_samples = int(self.n_samples * self.ensemble_samples_coeff)
                term_samples = max(2, term_samples)
                ensemble_terms.add_term(model_params, self.iqp_circuit, self.sigma, self.n_ops,
                                       term_samples, step_key_inner,
                                       init_coefs=self.init_coefs, wires=self.wires,
                                       indep_estimates=self.indep_estimates, verbose=verbose)
                step_key_inner, _ = jax.random.split(step_key_inner)
            # Update weights for the fresh terms
            ensemble_terms.trs = jnp.array(ensemble_terms.trs)
            ensemble_terms.corrs = jnp.array(ensemble_terms.corrs)
        
        loss_kwargs = {
            "params": params_init,
            "iqp_circuit": self.iqp_circuit,
            "weights": self.weights,
            "ground_truth": ground_truth,
            "ensemble_terms": ensemble_terms,
            "sigma": self.sigma,
            "n_ops": self.n_ops,
            "n_samples": self.n_samples,
            "lambda_dual": self.lambda_dual,
            "key": step_key,
            "verbose": verbose
        }

        trainer = iqp.Trainer("Adam", dual_mmd_loss, stepsize=stepsize)
        trainer.train(n_iters=steps, loss_kwargs=loss_kwargs, monitor_interval=monitor_interval)
        
        # Compute exact components from params_hist (use ALL checkpoints for full resolution)
        data_losses = []
        ensemble_losses = []
        sampled_epochs = []
        total_losses = np.array(trainer.losses)
        
        if hasattr(trainer, 'params_hist') and trainer.params_hist and len(trainer.params_hist) > 0:
            params_hist_list = list(trainer.params_hist) if isinstance(trainer.params_hist, jnp.ndarray) else trainer.params_hist
            
            # Compute components at ALL parameter checkpoints
            if monitor_interval is not None:
                if verbose:
                    print(f"  Computing loss components for {len(params_hist_list)} checkpoints...")
                for idx, params_at_epoch in enumerate(params_hist_list):
                    epoch_num = idx * monitor_interval
                    sampled_epochs.append(epoch_num)
                    
                    # Create fresh key for each evaluation
                    eval_key = jax.random.fold_in(step_key, epoch_num)
                    
                    components = dual_mmd_loss(
                        params_at_epoch, self.iqp_circuit, ground_truth, ensemble_terms,
                        self.weights, self.sigma, self.n_ops, self.n_samples, eval_key,
                        init_coefs=self.init_coefs, wires=self.wires,
                        indep_estimates=self.indep_estimates, lambda_dual=self.lambda_dual,
                        return_components=True
                    )
                    data_losses.append(float(components['data']))
                    ensemble_losses.append(float(components['ensemble']))
        else:
            if verbose:
                print(f"  Warning: No params_hist available, using final values only")
            sampled_epochs = [steps - 1]
            # Use final params
            components = dual_mmd_loss(
                trainer.final_params, self.iqp_circuit, ground_truth, ensemble_terms,
                self.weights, self.sigma, self.n_ops, self.n_samples, step_key,
                init_coefs=self.init_coefs, wires=self.wires,
                indep_estimates=self.indep_estimates, lambda_dual=self.lambda_dual,
                return_components=True
            )
            data_losses.append(float(components['data']))
            ensemble_losses.append(float(components['ensemble']))
        
        # Compute final components for storage
        final_components = dual_mmd_loss(
            trainer.final_params, self.iqp_circuit, ground_truth, ensemble_terms,
            self.weights, self.sigma, self.n_ops, self.n_samples, step_key,
            init_coefs=self.init_coefs, wires=self.wires, 
            indep_estimates=self.indep_estimates, lambda_dual=self.lambda_dual,
            return_components=True
        )
        
        # Store training history with component trajectories
        self.training_losses.append({
            'total': total_losses,
            'data_final': float(final_components['data']),
            'ensemble_final': float(final_components['ensemble']),
            'data_history': np.array(data_losses),
            'ensemble_history': np.array(ensemble_losses),
            'sampled_epochs': np.array(sampled_epochs)
        })
        
        if verbose:
            print(f"  Captured {len(data_losses)} component evaluations (at epochs {sampled_epochs[0]}, ..., {sampled_epochs[-1]})")
            print(f"  Final: data_loss={final_components['data']:.6f}, ens_loss={final_components['ensemble']:.6f}, dual={total_losses[-1]:.6f}")

        key, subkey = jax.random.split(key, 2)
        self.add_model(trainer.final_params, subkey)
        return key
