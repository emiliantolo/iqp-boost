"""Best-config retraining for completed HPO studies."""

from __future__ import annotations

import copy
import json
import math
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from src.experiments.factory import build_dataset_bundle
from src.run import run_boosting_experiment


@dataclass(frozen=True)
class BestRetrainSpec:
    dataset_spec: dict
    plot_spec: dict
    n_seeds: int = 5
    seed_start: int = 0
    baseline: str = "standalone"
    report_fcfw: bool = True
    skip_sampling: bool = True
    final_eval_sampling: bool = True
    output_subdir: str = "best_retrains"
    config_overrides: dict[str, Any] | None = None
    n_jobs: int = 1


def json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def resolve_best_retrain_spec(hpo_spec: dict) -> BestRetrainSpec | None:
    """Resolve Best retrain settings from a raw HPO spec."""
    retrain_spec = hpo_spec.get("best_retrains")
    if retrain_spec is None:
        return None
    # Best retrains run after HPO finalization and can involve JAX-heavy
    # shutdown paths. Keep them sequential unless explicitly configured.
    return BestRetrainSpec(
        dataset_spec=hpo_spec["dataset"],
        plot_spec=hpo_spec.get("plot", {"kind": "none"}),
        n_seeds=int(retrain_spec.get("n_seeds", 5)),
        seed_start=int(retrain_spec.get("seed_start", 0)),
        baseline=retrain_spec.get("baseline", "standalone"),
        report_fcfw=bool(retrain_spec.get("report_fcfw", True)),
        skip_sampling=bool(retrain_spec.get("skip_sampling", True)),
        final_eval_sampling=bool(retrain_spec.get("final_eval_sampling", True)),
        output_subdir=retrain_spec.get("output_subdir", "best_retrains"),
        config_overrides=copy.deepcopy(retrain_spec.get("config_overrides", {})),
        n_jobs=int(retrain_spec.get("n_jobs", 1)),
    )


def _run_single_retrain(
    run_config: dict,
    seed_idx: int,
    seed: int,
    dataset_spec: dict,
    plot_spec: dict,
    output_dir: str,
    hpo_dir: str,
    log_filename: str,
    baseline: str,
    report_fcfw: bool,
    skip_sampling: bool,
    final_eval_sampling: bool,
) -> dict:
    """Self-contained retrain worker intended for ``spawn``-ed processes.

    Each worker runs a single seed end-to-end, saves the model, and returns a
    JSON-serializable payload.  JAX is isolated per-process so there is no
    risk of compilation-cache corruption.
    """
    run_config["rng_seed"] = seed
    run_config["data_seed"] = seed
    run_config["baseline"] = baseline
    run_config["report_fcfw"] = report_fcfw
    run_config["skip_sampling"] = skip_sampling
    run_config["final_eval_sampling"] = final_eval_sampling

    bundle = build_dataset_bundle(
        dataset_spec=dataset_spec,
        config=run_config,
        plot_spec=plot_spec,
    )
    run_name = f"seed_{seed_idx:03d}"
    result = run_boosting_experiment(
        config=run_config,
        dataset=bundle,
        dataset_spec=dataset_spec,
        output_base_dir=output_dir,
        run_name=run_name,
        log_dir=hpo_dir,
        log_filename=log_filename,
        append_log=True,
        skip_plots=False,
    )
    run_dir = Path(result["output_dir"])
    model_path = run_dir / "ensemble.npz"
    result["ensemble"].save(str(model_path))
    final_stats = result["final_stats"]
    fcfw_stats = result.get("ensemble_fcfw_stats")
    payload = {
        "seed_index": seed_idx,
        "seed": seed,
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "final_stats": final_stats,
        "baseline_stats": result.get("baseline_stats"),
        "ensemble_fcfw_stats": fcfw_stats,
        "n_models_accepted": int(result["n_models_accepted"]),
        "weights": np.asarray(result["weights"], dtype=np.float64).tolist(),
    }
    fcfw_weights = result.get("ensemble_fcfw_weights")
    if fcfw_weights is not None:
        payload["ensemble_fcfw_weights"] = np.asarray(fcfw_weights, dtype=np.float64).tolist()

    # Artifact paths for post-hoc analysis
    payload["artifacts"] = {
        "ensemble": str(model_path),
        "baseline_artifacts": str(run_dir / "baseline_artifacts.npz") if (run_dir / "baseline_artifacts.npz").exists() else None,
        "data_only_ensemble": str(run_dir / "data_only_ensemble.npz") if (run_dir / "data_only_ensemble.npz").exists() else None,
        "samples": str(run_dir / "samples.npz") if (run_dir / "samples.npz").exists() else None,
        "results_csv": str(run_dir / "results.csv") if (run_dir / "results.csv").exists() else None,
        "config": str(run_dir / "config.json") if (run_dir / "config.json").exists() else None,
    }

    (run_dir / "final_stats.json").write_text(
        json.dumps(payload, indent=2, default=json_default)
    )
    return payload


def run_best_retrains(
    spec: BestRetrainSpec,
    best_config_path: Path | None,
    hpo_dir: Path,
) -> dict | None:
    """Retrain the winning HPO config for multiple seeds and aggregate metrics."""
    if best_config_path is None or not best_config_path.exists():
        print("[Best retrains] best_config.json missing, skipping retrains")
        return None

    output_dir = hpo_dir / spec.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    best_config = json.loads(best_config_path.read_text())
    per_seed = []

    n_workers = min(spec.n_jobs, spec.n_seeds)
    if n_workers > 1:
        print(f"[Best retrains] Running {spec.n_seeds} seeds across {n_workers} workers")
        # Build argument list for starmap.
        args_list = []
        for seed_idx in range(spec.n_seeds):
            seed = spec.seed_start + seed_idx
            run_config = copy.deepcopy(best_config)
            run_config = _deep_merge(run_config, spec.config_overrides or {})
            args_list.append(
                (
                    run_config,
                    seed_idx,
                    seed,
                    spec.dataset_spec,
                    spec.plot_spec,
                    str(output_dir),
                    str(hpo_dir),
                    "best_retrains.log",
                    spec.baseline,
                    spec.report_fcfw,
                    spec.skip_sampling,
                    spec.final_eval_sampling,
                )
            )

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            per_seed = pool.starmap(_run_single_retrain, args_list)
    else:
        for seed_idx in range(spec.n_seeds):
            seed = spec.seed_start + seed_idx
            run_config = copy.deepcopy(best_config)
            run_config = _deep_merge(run_config, spec.config_overrides or {})
            payload = _run_single_retrain(
                run_config,
                seed_idx,
                seed,
                spec.dataset_spec,
                spec.plot_spec,
                str(output_dir),
                str(hpo_dir),
                "best_retrains.log",
                spec.baseline,
                spec.report_fcfw,
                spec.skip_sampling,
                spec.final_eval_sampling,
            )
            per_seed.append(payload)

    summary = {
        "n_seeds": spec.n_seeds,
        "baseline": spec.baseline,
        "report_fcfw": spec.report_fcfw,
        "seeds": per_seed,
        "aggregates": _aggregate_seed_metrics(per_seed),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default))

    # Write artifact manifest for easy post-hoc discovery
    manifest = {
        "n_seeds": spec.n_seeds,
        "summary": str(output_dir / "summary.json"),
        "seeds": [
            {
                "seed_index": seed.get("seed_index"),
                "seed": seed.get("seed"),
                "run_dir": seed.get("run_dir"),
                "artifacts": seed.get("artifacts", {}),
            }
            for seed in per_seed
        ],
    }
    (output_dir / "retrain_artifacts.json").write_text(json.dumps(manifest, indent=2, default=json_default))
    return summary


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copied base dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _aggregate_seed_metrics(per_seed: list[dict]) -> dict:
    groups = {
        "final_stats": [item.get("final_stats") for item in per_seed],
        "baseline_stats": [item.get("baseline_stats") for item in per_seed],
        "ensemble_fcfw_stats": [item.get("ensemble_fcfw_stats") for item in per_seed],
    }
    return {name: _aggregate_dicts(values) for name, values in groups.items()}


def _aggregate_dicts(values: list[dict | None]) -> dict:
    keys = set()
    for value in values:
        if isinstance(value, dict):
            keys.update(value.keys())

    aggregate = {}
    for key in sorted(keys):
        numeric = []
        for value in values:
            if not isinstance(value, dict):
                continue
            item = value.get(key)
            if isinstance(item, (int, float)) and math.isfinite(float(item)):
                numeric.append(float(item))
        if numeric:
            aggregate[key] = {
                "mean": float(mean(numeric)),
                "std": float(pstdev(numeric)) if len(numeric) > 1 else 0.0,
            }
    return aggregate
