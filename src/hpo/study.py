"""HPO study lifecycle for config-driven IQP boosting experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import optuna

from src.experiments.suite import DEFAULT_RUN_CONFIG
from src.hpo.finalization import finalize_study, write_no_completed_trials_summary
from src.hpo.spec import load_hpo_spec
from src.hpo.trial import HpoTrialContext, run_trial
from src.hpo.trial_config import deep_merge


def run_hpo(config_path: Path) -> optuna.study.Study:
    """Run an Optuna HPO study and persist best-trial artifacts."""
    spec = load_hpo_spec(config_path)
    objective_spec = spec.objective

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hpo_dir = spec.output_base / f"{spec.study_name}_{timestamp}"
    trials_dir = hpo_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    storage = spec.storage
    if storage is None:
        storage = f"sqlite:///{hpo_dir / 'study.db'}"

    base_config = deep_merge(DEFAULT_RUN_CONFIG, spec.fixed_config)

    study = optuna.create_study(
        study_name=spec.study_name,
        direction=spec.direction,
        sampler=optuna.samplers.TPESampler(seed=spec.sampler_seed),
        pruner=optuna.pruners.NopPruner(),
        storage=storage,
        load_if_exists=True,
    )

    trial_context = HpoTrialContext(
        base_config=base_config,
        search_space=spec.search_space,
        dataset_spec=spec.dataset_spec,
        plot_spec=spec.plot_spec,
        objective=objective_spec,
        trials_dir=trials_dir,
        hpo_dir=hpo_dir,
        metric_configs=spec.metric_configs,
        baseline_epochs=spec.baseline_epochs,
    )

    study.optimize(lambda trial: run_trial(trial, trial_context), n_trials=spec.n_trials)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        write_no_completed_trials_summary(
            study=study,
            hpo_spec=spec.raw,
            config_path=spec.config_path,
            hpo_dir=hpo_dir,
            objective_metric=objective_spec.requested_metric,
        )
        return study
    finalize_study(
        study=study,
        hpo_spec=spec.raw,
        config_path=spec.config_path,
        hpo_dir=hpo_dir,
        objective_metric=objective_spec.requested_metric,
    )
    return study
