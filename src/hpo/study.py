"""HPO study lifecycle for config-driven IQP boosting experiments."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import optuna

from src.experiments.suite import DEFAULT_RUN_CONFIG
from src.hpo.best_model_evaluation import evaluate_best_model
from src.hpo.best_retrains import run_best_retrains
from src.hpo.objective import resolve_objective_spec
from src.hpo.pruning import build_pruning_spec
from src.hpo.trial import HpoTrialContext, run_trial
from src.hpo.trial_config import deep_merge


def load_hpo_config(path: Path) -> dict:
    """Load a JSON or TOML HPO spec."""
    if not path.exists():
        raise FileNotFoundError(f"HPO config not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() == ".toml":
        import tomllib

        return tomllib.loads(path.read_text())
    raise ValueError("Unsupported HPO config format. Use .json or .toml")


def json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def run_hpo(config_path: Path) -> optuna.study.Study:
    """Run an Optuna HPO study and persist best-trial artifacts."""
    hpo_spec = load_hpo_config(config_path)
    study_name = hpo_spec.get("study_name", config_path.stem)
    n_trials = int(hpo_spec.get("n_trials", 60))
    sampler_seed = int(hpo_spec.get("sampler_seed", 42))
    objective_spec = resolve_objective_spec(hpo_spec)

    output_base = Path(hpo_spec.get("output_dir", "out/hpo"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hpo_dir = output_base / f"{study_name}_{timestamp}"
    trials_dir = hpo_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    storage = hpo_spec.get("storage")
    if storage is None:
        storage = f"sqlite:///{hpo_dir / 'study.db'}"

    base_config = deep_merge(DEFAULT_RUN_CONFIG, hpo_spec.get("fixed_config", {}))
    pruning = build_pruning_spec(hpo_spec, base_config)

    study = optuna.create_study(
        study_name=study_name,
        direction=hpo_spec.get("direction", "minimize"),
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=pruning.pruner,
        storage=storage,
        load_if_exists=True,
    )

    dataset_spec = hpo_spec["dataset"]
    plot_spec = hpo_spec.get("plot", {})
    search_space = hpo_spec.get("search_space", {})
    trial_context = HpoTrialContext(
        base_config=base_config,
        search_space=search_space,
        dataset_spec=dataset_spec,
        plot_spec=plot_spec,
        objective=objective_spec,
        pruning=pruning,
        trials_dir=trials_dir,
        hpo_dir=hpo_dir,
        metric_configs=hpo_spec.get("metric_configs"),
        baseline_epochs=hpo_spec.get("baseline_epochs"),
    )

    study.optimize(lambda trial: run_trial(trial, trial_context), n_trials=n_trials)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        write_no_completed_trials_summary(
            study=study,
            hpo_spec=hpo_spec,
            config_path=config_path,
            hpo_dir=hpo_dir,
            objective_metric=objective_spec.requested_metric,
        )
        return study
    finalize_study(
        study=study,
        hpo_spec=hpo_spec,
        config_path=config_path,
        hpo_dir=hpo_dir,
        objective_metric=objective_spec.requested_metric,
    )
    return study


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
        "message": "No completed trials; all trials were pruned or failed.",
    }
    (hpo_dir / "study_summary.json").write_text(json.dumps(summary, indent=2, default=json_default))
    (hpo_dir / "hpo_config.json").write_text(json.dumps(hpo_spec, indent=2, default=json_default))
    print("No completed trials; skipping best-model artifact finalization.")
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

    retrain_summary = run_best_retrains(
        hpo_spec,
        best_config_path=best_config_path,
        hpo_dir=hpo_dir,
    )
    if retrain_summary is not None:
        summary["best_retrains"] = {
            "summary_path": str(hpo_dir / "best_retrains" / "summary.json"),
            "aggregates": retrain_summary.get("aggregates", {}),
        }

    (hpo_dir / "study_summary.json").write_text(json.dumps(summary, indent=2, default=json_default))
    (hpo_dir / "hpo_config.json").write_text(json.dumps(hpo_spec, indent=2, default=json_default))

    print(f"Best trial: {best.number} {objective_metric.upper()}={best.value:.6f}")
    print(f"Best model saved to: {best_model_dst}")
    return summary
