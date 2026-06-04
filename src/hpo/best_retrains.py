"""Best-config retraining for completed HPO studies."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from src.experiments.factory import build_dataset_bundle
from src.runner import run_boosting_experiment


def json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def run_best_retrains(
    hpo_spec: dict,
    best_config_path: Path | None,
    hpo_dir: Path,
) -> dict | None:
    """Retrain the winning HPO config for multiple seeds and aggregate metrics."""
    retrain_spec = hpo_spec.get("best_retrains")
    if not retrain_spec:
        return None
    if best_config_path is None or not best_config_path.exists():
        print("[Best retrains] best_config.json missing, skipping retrains")
        return None

    n_seeds = int(retrain_spec.get("n_seeds", 5))
    seed_start = int(retrain_spec.get("seed_start", 0))
    baseline = retrain_spec.get("baseline", "standalone")
    report_fcfw = bool(retrain_spec.get("report_fcfw", True))
    output_dir = hpo_dir / retrain_spec.get("output_subdir", "best_retrains")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_config = json.loads(best_config_path.read_text())
    per_seed = []
    for seed_idx in range(n_seeds):
        seed = seed_start + seed_idx
        run_config = copy.deepcopy(best_config)
        run_config["rng_seed"] = seed
        run_config["data_seed"] = seed
        run_config["baseline"] = baseline
        run_config["report_fcfw"] = report_fcfw
        run_config["skip_sampling"] = bool(retrain_spec.get("skip_sampling", True))
        run_config["final_eval_sampling"] = bool(retrain_spec.get("final_eval_sampling", True))

        bundle = build_dataset_bundle(
            dataset_spec=hpo_spec["dataset"],
            config=run_config,
            plot_spec=hpo_spec.get("plot", {"kind": "none"}),
        )
        run_name = f"seed_{seed_idx:03d}"
        run_kwargs = {
            "config": run_config,
            "dataset_name": bundle["dataset_name"],
            "dataset_spec": hpo_spec["dataset"],
            "x_train": bundle["x_train"],
            "validity_fn": bundle["validity_fn"],
            "coverage_fn": bundle["coverage_fn"],
            "custom_viz_fn": bundle["custom_viz_fn"],
            "top_k_tvd_fn": bundle.get("top_k_tvd_fn"),
            "exact_probs": bundle.get("exact_probs"),
            "generation_eval_fn": bundle.get("generation_eval_fn"),
            "output_base_dir": str(output_dir),
            "run_name": run_name,
            "log_dir": str(hpo_dir),
            "log_filename": "best_retrains.log",
            "append_log": True,
        }
        if "x_test" in bundle:
            run_kwargs["x_test"] = bundle["x_test"]

        result = run_boosting_experiment(**run_kwargs)
        run_dir = Path(result["output_dir"])
        model_path = run_dir / "ensemble.json"
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
        (run_dir / "final_stats.json").write_text(json.dumps(payload, indent=2, default=json_default))
        per_seed.append(payload)

    summary = {
        "n_seeds": n_seeds,
        "baseline": baseline,
        "report_fcfw": report_fcfw,
        "seeds": per_seed,
        "aggregates": _aggregate_seed_metrics(per_seed),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default))
    return summary


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
