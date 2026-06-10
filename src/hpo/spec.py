"""Resolved HPO spec loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.hpo.objective import ObjectiveSpec, resolve_objective_spec


ALLOWED_TOP_LEVEL_KEYS = {
    "study_name",
    "n_trials",
    "n_jobs",
    "sampler_seed",
    "objective_metric",
    "direction",
    "storage",
    "output_dir",
    "dataset",
    "plot",
    "fixed_config",
    "search_space",
    "metric_configs",
    "baseline_epochs",
    "best_retrains",
}

DICT_FIELDS = {
    "dataset",
    "plot",
    "fixed_config",
    "search_space",
    "best_retrains",
}


@dataclass(frozen=True)
class HpoSpec:
    raw: dict
    config_path: Path
    study_name: str
    n_trials: int
    n_jobs: int
    sampler_seed: int
    objective: ObjectiveSpec
    output_base: Path
    storage: str | None
    direction: str
    dataset_spec: dict
    plot_spec: dict
    fixed_config: dict
    search_space: dict
    metric_configs: list | None
    baseline_epochs: int | None


def load_hpo_spec(path: Path) -> HpoSpec:
    """Load, validate, and resolve a JSON or TOML HPO spec."""
    raw = _load_raw_config(path)
    _validate_hpo_spec(raw)
    return HpoSpec(
        raw=raw,
        config_path=path,
        study_name=str(raw.get("study_name", path.stem)),
        n_trials=int(raw.get("n_trials", 60)),
        n_jobs=int(raw.get("n_jobs", 1)),
        sampler_seed=int(raw.get("sampler_seed", 42)),
        objective=resolve_objective_spec(raw),
        output_base=Path(raw.get("output_dir", "out/hpo")),
        storage=raw.get("storage"),
        direction=str(raw.get("direction", "minimize")),
        dataset_spec=raw["dataset"],
        plot_spec=raw.get("plot", {}),
        fixed_config=raw.get("fixed_config", {}),
        search_space=raw.get("search_space", {}),
        metric_configs=raw.get("metric_configs"),
        baseline_epochs=raw.get("baseline_epochs"),
    )


def _load_raw_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"HPO config not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() == ".toml":
        import tomllib

        return tomllib.loads(path.read_text())
    raise ValueError("Unsupported HPO config format. Use .json or .toml")


def _validate_hpo_spec(raw: Any) -> None:
    if not isinstance(raw, dict):
        raise ValueError("HPO config must be a JSON/TOML object")

    unknown = sorted(set(raw) - ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"Unsupported HPO config keys: {', '.join(unknown)}")

    if "dataset" not in raw:
        raise ValueError("HPO config requires dataset")

    for field in sorted(DICT_FIELDS):
        if field in raw and not isinstance(raw[field], dict):
            raise ValueError(f"HPO config field {field!r} must be a dict")
