"""Baseline lifecycle for IQP boosting experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import iqpopt as iqp
import jax
import numpy as np
from iqpopt.gen_qml.iqp_methods import mmd_loss_iqp

from src.core import BoostedEnsemble, EvaluationPolicy, get_params_init
from src.datasets import DatasetBundle
from src.run.boosting_step import BoostingStepContext, initialize_boosting_ensemble, run_boosting_step
from src.io.reporting import report_baseline


@dataclass(frozen=True)
class BaselineContext:
    config: dict
    circuit: iqp.IqpSimulator
    dataset: DatasetBundle
    x_train: np.ndarray
    key: jax.Array
    sigma: float | list
    n_ops: int
    n_samples: int
    wires: list | None
    monitor_interval: int | None
    turbo_opt: int | None
    evaluation: EvaluationPolicy
    acceptance_metric: str
    min_alpha_accept: float
    baseline_epochs: int | None = None


@dataclass
class BaselineResult:
    key: jax.Array
    selected_baselines: list[str]
    standalone_stats: dict | None = None
    standalone_samples: np.ndarray | None = None
    standalone_params: object | None = None
    standalone_train_losses: object | None = None
    data_only_ensemble: BoostedEnsemble | None = None
    data_only_stats: dict | None = None
    data_only_history: dict | None = None
    reference_label: str | None = None
    reference_stats: dict | None = None

    def reference_or_synthetic(self, m0_training_mmd: float) -> tuple[str, dict]:
        if self.reference_label is not None and self.reference_stats is not None:
            return self.reference_label, self.reference_stats
        return "Synthetic", {"mmd": m0_training_mmd}


def run_baselines(context: BaselineContext) -> BaselineResult:
    """Train configured baselines and select the final-evaluation reference."""
    selected = resolve_baselines_to_run(context.config)
    print(f"Baselines enabled: {', '.join(selected)}")

    key = context.key
    standalone_stats = None
    standalone_samples = None
    standalone_params = None
    standalone_train_losses = None
    data_only_ensemble = None
    data_only_stats = None
    data_only_history = None

    if "standalone" in selected:
        key, standalone_samples, standalone_stats, standalone_params, standalone_train_losses = _run_standalone_baseline(
            context,
            key,
        )

    if "data_only" in selected:
        key, data_only_ensemble, data_only_history, data_only_stats = _run_data_only_ensemble_baseline(context, key)
        print(f"Data-only iterative baseline final MMD={data_only_stats['mmd']:.4f}")
        report_baseline(data_only_stats["mmd"], data_only_stats)

    reference_label, reference_stats = _select_reference(
        selected,
        standalone_stats=standalone_stats,
        data_only_stats=data_only_stats,
    )

    return BaselineResult(
        key=key,
        selected_baselines=selected,
        standalone_stats=standalone_stats,
        standalone_samples=standalone_samples,
        standalone_params=standalone_params,
        standalone_train_losses=standalone_train_losses,
        data_only_ensemble=data_only_ensemble,
        data_only_stats=data_only_stats,
        data_only_history=data_only_history,
        reference_label=reference_label,
        reference_stats=reference_stats,
    )


def resolve_baselines_to_run(config: dict) -> list[str]:
    """Resolve enabled baseline names from current and legacy config keys."""
    if "baseline" in config:
        baselines = config.get("baseline")
        if baselines is None:
            return []
    else:
        baselines = config.get("baselines_to_run", None)
        if baselines is None:
            baselines = ["standalone"]
            if config.get("run_data_only_baseline", False):
                baselines.append("data_only")

    if isinstance(baselines, str):
        baselines = [baselines]
    baselines = [b for b in baselines if b in {"standalone", "data_only", "none"}]
    if "none" in baselines:
        return []
    if not baselines:
        return ["standalone"]
    return baselines


def _run_standalone_baseline(
    context: BaselineContext,
    key: jax.Array,
) -> tuple[jax.Array, np.ndarray | None, dict, object, object]:
    baseline_epochs = context.baseline_epochs
    if baseline_epochs is None:
        baseline_epochs = context.config["epochs_per_step"] * context.config["n_models"]

    print(f"\nTraining standalone baseline model ({baseline_epochs} epochs)...")
    key, trainer, params = train_standalone_model(context, key, baseline_epochs)

    train_losses = getattr(trainer, "losses", [])
    final_loss = float(train_losses[-1]) if len(train_losses) > 0 else float("nan")

    if not context.evaluation.final_sampling_enabled:
        print(
            "  [skip_sampling=True] Skipping baseline state vector evaluation. "
            f"Using training loss: {final_loss:.6f}"
        )
        model_probs = (
            context.evaluation.exact_model_probs(context.circuit, params, context.wires)
            if hasattr(context.evaluation, "exact_metrics_enabled") and context.evaluation.exact_metrics_enabled("baseline")
            else None
        )
        stats = context.evaluation.analytical_mmd_stats(final_loss, model_probs=model_probs)
        stats["training_loss"] = final_loss
        report_baseline(final_loss, stats)
        return key, None, stats, params, train_losses

    samples, stats = context.evaluation.sample_and_evaluate_circuit(
        context.circuit,
        params,
        wires=context.wires,
    )
    stats["training_loss"] = final_loss
    print()
    report_baseline(stats["mmd"], stats)
    print(f"  (Analytical baseline MMD from training: {final_loss:.6f})")
    return key, samples, stats, params, train_losses


def train_standalone_model(
    context: BaselineContext,
    key: jax.Array,
    epochs: int,
) -> tuple[jax.Array, object, object]:
    """Train a single standalone IQP model using iqpopt's MMD loss."""
    key, init_key = jax.random.split(key, 2)
    params_init = get_params_init(context.config.get("init_baseline", "random"), context.circuit, context.x_train, init_key)

    loss_kwargs = {
        "params": params_init,
        "iqp_circuit": context.circuit,
        "ground_truth": context.x_train,
        "sigma": context.sigma,
        "n_ops": context.n_ops,
        "n_samples": context.n_samples,
        "wires": context.wires,
        "max_batch_ops": context.config.get("max_batch_ops", None),
        "max_batch_samples": context.config.get("max_batch_samples", None),
    }

    trainer = iqp.Trainer("Adam", mmd_loss_iqp, stepsize=context.config["learning_rate"])
    trainer.train(
        n_iters=epochs,
        loss_kwargs=loss_kwargs,
        monitor_interval=context.monitor_interval,
        turbo=context.turbo_opt,
    )

    return key, trainer, trainer.final_params


def _run_data_only_ensemble_baseline(
    context: BaselineContext,
    key: jax.Array,
) -> tuple[jax.Array, BoostedEnsemble, dict, dict]:
    n_models = int(context.config["n_models"])
    cfg = dict(context.config)
    cfg["lambda_dual"] = 0.0

    print(f"\nTraining data-only iterative baseline ({n_models} models, lambda_dual=0)...")

    ensemble = BoostedEnsemble(
        context.circuit,
        n_models=n_models,
        sigma=context.sigma,
        n_ops=context.n_ops,
        n_samples=context.n_samples,
        lambda_dual=0.0,
        wires=context.wires,
        max_batch_ops=context.config.get("max_batch_ops", None),
        max_batch_samples=context.config.get("max_batch_samples", None),
    )

    initial = initialize_boosting_ensemble(
        ensemble=ensemble,
        x_train=context.x_train,
        key=key,
        config=cfg,
        monitor_interval=context.monitor_interval,
        turbo_opt=context.turbo_opt,
        evaluation=context.evaluation,
        print_model0_loss=False,
        include_sampled_tvd=False,
    )
    key = initial.key
    history = initial.metrics_history
    prev_stats = initial.stats
    prev_training_mmd = initial.training_mmd

    for step in range(1, n_models):
        print(f"[Data-only Step {step}] Training Model {step}...")
        step_result = run_boosting_step(
            BoostingStepContext(
                config=cfg,
                ensemble=ensemble,
                x_train=context.x_train,
                key=key,
                evaluation=context.evaluation,
                metrics_history=history,
                step=step,
                monitor_interval=context.monitor_interval,
                turbo_opt=context.turbo_opt,
                acceptance_metric=context.acceptance_metric,
                min_alpha_accept=context.min_alpha_accept,
                previous_stats=prev_stats,
                previous_training_mmd=prev_training_mmd,
                validity_fn=context.dataset.validity_fn,
                coverage_fn=context.dataset.coverage_fn,
                step_prefix="Data-only Step",
                include_sampled_tvd=False,
                apply_lambda_schedule=False,
                compute_snr=False,
                run_cleanup=False,
            )
        )
        key = step_result.key
        if step_result.should_stop:
            break
        prev_stats = step_result.previous_stats
        prev_training_mmd = step_result.previous_training_mmd

    if not context.evaluation.final_sampling_enabled:
        final_stats = context.evaluation.analytical_mmd_stats(
            history["training_loss"][-1] if history["training_loss"] else float("nan")
        )
    else:
        _, final_stats = context.evaluation.sample_and_evaluate_ensemble(ensemble, step=n_models)

    return key, ensemble, history, final_stats


def _select_reference(
    selected: list[str],
    *,
    standalone_stats: dict | None,
    data_only_stats: dict | None,
) -> tuple[str | None, dict | None]:
    catalog = {
        "standalone": ("Standalone", standalone_stats),
        "data_only": ("Data-only", data_only_stats),
    }
    for name in selected:
        label, stats = catalog.get(name, (None, None))
        if stats is not None:
            print(f"Baseline used as reference: {label}")
            return label, stats
    return None, None


def save_baseline_artifacts(result: BaselineResult, output_dir: Path) -> None:
    """Save baseline training curves and parameters for post-hoc analysis."""
    stl = result.standalone_train_losses
    artifacts = {
        "standalone_train_losses": np.array(stl if stl is not None else [], dtype=np.float64),
    }
    if result.standalone_params is not None:
        artifacts["standalone_params"] = np.array(result.standalone_params)
    if result.standalone_samples is not None:
        artifacts["standalone_samples"] = result.standalone_samples
    if result.data_only_ensemble is not None:
        result.data_only_ensemble.save(str(output_dir / "data_only_ensemble.npz"))
    if result.data_only_history is not None:
        artifacts["data_only_history"] = json.dumps(result.data_only_history)
    if result.data_only_stats is not None:
        artifacts["data_only_stats"] = json.dumps(result.data_only_stats)
    if result.standalone_stats is not None:
        artifacts["standalone_stats"] = json.dumps(result.standalone_stats)
    np.savez_compressed(output_dir / "baseline_artifacts.npz", **artifacts)
