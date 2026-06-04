"""Objective metric resolution and validation for HPO."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveSpec:
    requested_metric: str
    final_metric_key: str

    @property
    def is_intermediate_available(self) -> bool:
        return self.requested_metric not in {"exact_tvd", "test_mmd"}

    def final_value(self, final_stats: dict, trial_number: int) -> float:
        metric_value = float(final_stats.get(self.final_metric_key, float("nan")))
        if not math.isfinite(metric_value):
            raise ValueError(
                f"Trial {trial_number} produced invalid {self.requested_metric}: {metric_value}"
            )
        return metric_value


def resolve_objective_spec(hpo_spec: dict) -> ObjectiveSpec:
    metric = hpo_spec.get("objective_metric", "tvd")
    if metric == "exact_tvd":
        return ObjectiveSpec(requested_metric="exact_tvd", final_metric_key="tvd")
    return ObjectiveSpec(requested_metric=metric, final_metric_key=metric)


def validate_objective_before_training(objective: ObjectiveSpec, run_config: dict, bundle: dict) -> None:
    if objective.requested_metric == "exact_tvd":
        if not bool(run_config.get("exact_sampling", False)):
            raise ValueError("objective_metric=exact_tvd requires fixed_config.exact_sampling=true")
        if bool(run_config.get("skip_sampling", False)) and not bool(run_config.get("final_eval_sampling", False)):
            raise ValueError(
                "objective_metric=exact_tvd requires final sampling via skip_sampling=false "
                "or final_eval_sampling=true"
            )
        run_config["require_exact_sampling"] = True

    if objective.requested_metric == "test_mmd" and "x_test" not in bundle:
        raise ValueError("objective_metric=test_mmd requires the dataset bundle to provide x_test")
