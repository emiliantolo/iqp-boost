"""Trial configuration resolution for Optuna-backed HPO."""

from __future__ import annotations

import copy
import math
from typing import Any

import optuna


def deep_merge(base: dict, override: dict) -> dict:
    """Return a deep copy of ``base`` with nested ``override`` values applied."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_nested(target: dict, dotted_key: str, value: Any) -> None:
    """Set ``target[a][b]`` for dotted key ``a.b``, creating dictionaries."""
    if not dotted_key or any(part == "" for part in dotted_key.split(".")):
        raise ValueError(f"Invalid dotted config key: {dotted_key!r}")

    current = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is not None and not isinstance(existing, dict):
            raise ValueError(f"Cannot set nested key {dotted_key!r}: {part!r} is not a dict")
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def sample_param(trial: optuna.Trial, name: str, spec: dict) -> Any:
    """Sample one configured parameter from an Optuna trial."""
    if not isinstance(spec, dict):
        raise ValueError(f"Search-space spec for {name} must be a dict")

    kind = spec.get("type")
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if kind == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
    if kind == "categorical":
        choices = spec.get("choices")
        if choices is None:
            raise ValueError(f"Categorical search-space spec for {name} requires choices")
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"Unsupported search-space type for {name}: {kind}")


def bodyness_to_sigma(k: float, n_qubits: int) -> float:
    """Convert expected Pauli bodyness target to Gaussian kernel bandwidth."""
    n_qubits = int(n_qubits)
    k = float(k)
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive for bodyness sigma sampling")
    if not 0.0 < k < n_qubits / 2:
        raise ValueError(f"bodyness target must be in (0, n_qubits / 2), got {k}")
    return float(math.sqrt(-1.0 / (2.0 * math.log(1.0 - 2.0 * k / n_qubits))))


def sample_bodyness_sigma(trial: optuna.Trial, name: str, spec: dict) -> tuple[list[float], dict]:
    """Sample an ordered, separated sigma vector through expected bodyness targets."""
    n_qubits = int(spec["n_qubits"])
    choices = spec.get("n_sigmas_choices", [1, 2, 3])
    n_sigmas = int(trial.suggest_categorical(f"{name}.n_sigmas", choices))
    if n_sigmas <= 0:
        raise ValueError("n_sigmas must be positive")

    low = float(spec.get("low", 0.5))
    high = float(spec.get("high", (n_qubits / 2.0) * (1.0 - 1e-5)))
    min_separation = float(spec.get("min_separation", 1.5))
    if not 0.0 < low < high < n_qubits / 2.0:
        raise ValueError("bodyness sigma range must satisfy 0 < low < high < n_qubits / 2")

    remaining_span = high - low - min_separation * (n_sigmas - 1)
    if remaining_span < 0:
        raise ValueError(
            "bodyness sigma range is too narrow for the requested n_sigmas and min_separation"
        )

    raw = [
        float(trial.suggest_float(f"{name}.bodyness_raw_{idx}", 0.0, 1.0))
        for idx in range(n_sigmas)
    ]
    raw_sorted = sorted(raw)
    targets = [
        low + idx * min_separation + raw_value * remaining_span
        for idx, raw_value in enumerate(raw_sorted)
    ]
    sigmas = [bodyness_to_sigma(k, n_qubits) for k in targets]
    metadata = {
        f"{name}.n_sigmas": n_sigmas,
        f"{name}.bodyness_targets": targets,
        f"{name}.values": sigmas,
    }
    return sigmas, metadata


def resolve_trial_config(base_config: dict, search_space: dict, trial: optuna.Trial) -> dict:
    """Resolve one runnable experiment config for an Optuna trial."""
    run_config = copy.deepcopy(base_config)
    sampled = {}
    for dotted_key, spec in search_space.items():
        if isinstance(spec, dict) and spec.get("type") == "bodyness_sigma":
            value, metadata = sample_bodyness_sigma(trial, dotted_key, spec)
            set_nested(run_config, dotted_key, value)
            run_config.pop("sigma_heuristic", None)
            run_config.pop("sigma_factor", None)
            sampled.update(metadata)
        else:
            value = sample_param(trial, dotted_key, spec)
            set_nested(run_config, dotted_key, value)
            sampled[dotted_key] = value
    trial.set_user_attr("sampled_config", sampled)
    return run_config
