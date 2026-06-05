"""One-trial execution for HPO studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna

from src.experiments.factory import build_dataset_bundle
from src.hpo.objective import ObjectiveSpec, validate_objective_before_training
from src.hpo.trial_config import resolve_trial_config
from src.run import run_boosting_experiment


@dataclass(frozen=True)
class HpoTrialContext:
    base_config: dict
    search_space: dict
    dataset_spec: dict
    plot_spec: dict
    objective: ObjectiveSpec
    trials_dir: Path
    hpo_dir: Path
    metric_configs: list | None = None
    baseline_epochs: int | None = None


def run_trial(trial: optuna.Trial, context: HpoTrialContext) -> float:
    """Run one Optuna trial from sampled config through persisted model attrs."""
    run_config = resolve_trial_config(context.base_config, context.search_space, trial)
    bundle = build_dataset_bundle(
        dataset_spec=context.dataset_spec,
        config=run_config,
        plot_spec=context.plot_spec,
    )
    validate_objective_before_training(context.objective, run_config, bundle)
    run_name = f"trial_{trial.number:04d}"

    result = run_boosting_experiment(
        config=run_config,
        dataset=bundle,
        dataset_spec=context.dataset_spec,
        metric_configs=context.metric_configs,
        baseline_epochs=context.baseline_epochs,
        output_base_dir=str(context.trials_dir),
        run_name=run_name,
        log_dir=str(context.hpo_dir),
        log_filename="hpo.log",
        append_log=True,
    )
    final_stats = result["final_stats"]
    metric_value = context.objective.final_value(final_stats, trial.number)

    output_dir = Path(result["output_dir"])
    model_path = output_dir / "ensemble.json"
    result["ensemble"].save(str(model_path))

    trial.set_user_attr("final_stats", final_stats)
    trial.set_user_attr("output_dir", str(output_dir))
    trial.set_user_attr("model_path", str(model_path))
    trial.set_user_attr("n_models_accepted", int(result["n_models_accepted"]))
    trial.set_user_attr("weights", np.asarray(result["weights"], dtype=np.float64).tolist())
    fcfw_weights = result.get("ensemble_fcfw_weights")
    if fcfw_weights is not None:
        trial.set_user_attr("ensemble_fcfw_weights", np.asarray(fcfw_weights, dtype=np.float64).tolist())
    return metric_value
