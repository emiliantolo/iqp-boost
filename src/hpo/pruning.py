"""Optuna pruner construction and intermediate reporting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import optuna

from src.hpo.objective import ObjectiveSpec


@dataclass(frozen=True)
class PruningSpec:
    pruner: optuna.pruners.BasePruner
    metric: str | None

    @property
    def enabled(self) -> bool:
        return self.metric is not None


def build_pruning_spec(hpo_spec: dict, base_config: dict) -> PruningSpec:
    spec = hpo_spec.get("pruner")
    if spec is None:
        return PruningSpec(pruner=optuna.pruners.NopPruner(), metric=None)
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict):
        raise ValueError("pruner must be a dict or string")

    kind = spec.get("type", "none")
    if kind in {None, "none", "nop"}:
        return PruningSpec(pruner=optuna.pruners.NopPruner(), metric=None)

    metric = spec.get("metric", "objective_metric")
    if kind == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(spec.get("n_startup_trials", 5)),
            n_warmup_steps=int(spec.get("n_warmup_steps", 0)),
            interval_steps=int(spec.get("interval_steps", 1)),
        )
    elif kind == "successive_halving":
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=_resolve_resource(spec.get("min_resource", 1), base_config, hpo_spec),
            reduction_factor=int(spec.get("reduction_factor", 3)),
            min_early_stopping_rate=int(spec.get("min_early_stopping_rate", 0)),
        )
    elif kind == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=_resolve_resource(spec.get("min_resource", 1), base_config, hpo_spec),
            max_resource=_resolve_resource(spec.get("max_resource", "n_models"), base_config, hpo_spec),
            reduction_factor=int(spec.get("reduction_factor", 3)),
        )
    else:
        raise ValueError(f"Unsupported pruner type: {kind}")

    return PruningSpec(pruner=pruner, metric=metric)


def make_trial_pruning_callback(
    trial: optuna.Trial,
    pruning: PruningSpec,
    objective: ObjectiveSpec,
) -> Any | None:
    if not pruning.enabled:
        return None

    def _callback(payload: dict) -> None:
        metric_name = pruning.metric
        if metric_name == "objective_metric":
            if not objective.is_intermediate_available:
                return
            metric_name = objective.final_metric_key
        value = payload.get(metric_name)
        if value is None:
            return
        value = float(value)
        if not math.isfinite(value):
            return
        step = int(payload["step"])
        trial.report(value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at step {step} on {metric_name}={value}")

    return _callback


def _resolve_resource(value, config: dict, hpo_spec: dict) -> int | str:
    if value == "n_models":
        search_spec = hpo_spec.get("search_space", {}).get("n_models")
        if isinstance(search_spec, dict):
            if search_spec.get("type") == "int":
                return int(search_spec["high"])
            if search_spec.get("type") == "categorical":
                return int(max(search_spec["choices"]))
        return int(config["n_models"])
    if value == "auto":
        return "auto"
    return int(value)
