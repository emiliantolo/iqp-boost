"""HPO study artifact finalization."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import optuna

from src.hpo.best_model_evaluation import evaluate_best_model
from src.hpo.best_retrains import resolve_best_retrain_spec, run_best_retrains
from src.hpo.retrain_plots import plot_best_retrain_summary


NO_COMPLETED_TRIALS_MESSAGE = "No completed trials; all trials failed or stopped before completion."


def json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def write_no_completed_trials_summary(
    study: optuna.study.Study,
    hpo_spec: dict,
    config_path: Path,
    hpo_dir: Path,
    objective_metric: str,
) -> dict:
    summary = {
        "study_name": study.study_name,
        "best_trial": None,
        "objective_metric": objective_metric,
        "best_value": None,
        "best_params": {},
        "best_user_attrs": {},
        "best_model_path": None,
        "best_config_path": None,
        "config_path": str(config_path),
        "message": NO_COMPLETED_TRIALS_MESSAGE,
    }
    (hpo_dir / "study_summary.json").write_text(json.dumps(summary, indent=2, default=json_default))
    (hpo_dir / "hpo_config.json").write_text(json.dumps(hpo_spec, indent=2, default=json_default))
    print(NO_COMPLETED_TRIALS_MESSAGE)
    return summary


def finalize_study(
    study: optuna.study.Study,
    hpo_spec: dict,
    config_path: Path,
    hpo_dir: Path,
    objective_metric: str,
) -> dict:
    """Persist best-trial artifacts and return the study summary."""
    best = study.best_trial

    best_model_src = Path(best.user_attrs["model_path"])
    best_model_dst = hpo_dir / "best_model.json"
    shutil.copyfile(best_model_src, best_model_dst)

    best_config_src = Path(best.user_attrs["output_dir"]) / "config.json"
    best_config_dst = hpo_dir / "best_config.json"
    best_config_path = None
    if best_config_src.exists():
        shutil.copyfile(best_config_src, best_config_dst)
        best_config_path = best_config_dst

    summary = {
        "study_name": study.study_name,
        "best_trial": best.number,
        "objective_metric": objective_metric,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "best_model_path": str(best_model_dst),
        "best_config_path": str(best_config_path) if best_config_path is not None else None,
        "config_path": str(config_path),
    }

    best_eval = evaluate_best_model(
        hpo_spec,
        best_config_path=best_config_path,
        best_model_path=best_model_dst,
        fcfw_weights_list=best.user_attrs.get("ensemble_fcfw_weights"),
    )
    if best_eval is not None:
        summary_key, metrics = best_eval
        summary[summary_key] = metrics
        if summary_key == "hamming_balls_metrics":
            metrics_path = hpo_dir / "best_hamming_balls_metrics.json"
        else:
            metrics_path = hpo_dir / f"best_{summary_key}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, default=json_default))
        print(f"Best model metrics saved to {metrics_path.name}")

    retrain_spec = resolve_best_retrain_spec(hpo_spec)
    retrain_summary = (
        run_best_retrains(retrain_spec, best_config_path=best_config_path, hpo_dir=hpo_dir)
        if retrain_spec is not None
        else None
    )
    if retrain_summary is not None:
        is_mnist = hpo_spec.get("dataset", {}).get("name") == "mnist"
        plot_paths = plot_best_retrain_summary(
            retrain_summary,
            hpo_dir,
            metric_filter="mmd" if is_mnist else None,
            include_weight_distribution=not is_mnist,
        )
        summary["best_retrains"] = {
            "summary_path": str(hpo_dir / "best_retrains" / "summary.json"),
            "aggregates": retrain_summary.get("aggregates", {}),
            "plots": plot_paths,
        }

    (hpo_dir / "study_summary.json").write_text(json.dumps(summary, indent=2, default=json_default))
    (hpo_dir / "hpo_config.json").write_text(json.dumps(hpo_spec, indent=2, default=json_default))

    print(f"Best trial: {best.number} {objective_metric.upper()}={best.value:.6f}")
    print(f"Best model saved to: {best_model_dst}")
    return summary
