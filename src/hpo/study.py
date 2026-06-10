"""HPO study lifecycle for config-driven IQP boosting experiments."""

from __future__ import annotations

import math
import multiprocessing
from datetime import datetime
from pathlib import Path

import optuna

from src.experiments.suite import DEFAULT_RUN_CONFIG
from src.hpo.finalization import finalize_study, write_no_completed_trials_summary
from src.hpo.spec import load_hpo_spec
from src.hpo.trial import HpoTrialContext, run_trial
from src.hpo.trial_config import deep_merge


def _make_sampler(spec):
    """Build an Optuna sampler from the HPO spec."""
    if spec.sampler == "tpe":
        return optuna.samplers.TPESampler(
            seed=spec.sampler_seed,
            multivariate=True,
            n_startup_trials=spec.n_startup_trials,
        )
    return optuna.samplers.GPSampler(
        seed=spec.sampler_seed,
        n_startup_trials=spec.n_startup_trials,
    )


def _run_hpo_worker(config_path: str, n_trials: int, storage: str, study_name: str,
                     direction: str, sampler_seed: int, sampler_name: str, n_startup_trials: int,
                     hpo_dir: str, trials_dir: str) -> None:
    """Self-contained worker that re-connects to the shared study and runs trials.

    This function is designed to be called inside a fresh ``spawn``-ed process
    so that each worker gets its own JAX runtime and the study storage backend
    handles trial-level coordination automatically.
    """
    # Re-import everything inside the worker so the process is self-contained.
    from src.hpo.spec import load_hpo_spec
    from src.hpo.trial import HpoTrialContext, run_trial
    from src.hpo.trial_config import deep_merge
    from src.experiments.suite import DEFAULT_RUN_CONFIG
    import optuna

    spec = load_hpo_spec(Path(config_path))
    base_config = deep_merge(DEFAULT_RUN_CONFIG, spec.fixed_config)
    objective_spec = spec.objective

    # Reconstruct sampler in the worker
    if sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(
            seed=sampler_seed,
            multivariate=True,
            n_startup_trials=n_startup_trials,
        )
    else:
        sampler = optuna.samplers.GPSampler(
            seed=sampler_seed,
            n_startup_trials=n_startup_trials,
        )

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
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
        trials_dir=Path(trials_dir),
        hpo_dir=Path(hpo_dir),
        metric_configs=spec.metric_configs,
        baseline_epochs=spec.baseline_epochs,
    )

    study.optimize(lambda trial: run_trial(trial, trial_context), n_trials=n_trials)


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
        if spec.n_jobs > 1:
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(str(hpo_dir / "journal.log"))
            )
        else:
            storage = f"sqlite:///{hpo_dir / 'study.db'}"

    base_config = deep_merge(DEFAULT_RUN_CONFIG, spec.fixed_config)

    study = optuna.create_study(
        study_name=spec.study_name,
        direction=spec.direction,
        sampler=_make_sampler(spec),
        pruner=optuna.pruners.NopPruner(),
        storage=storage,
        load_if_exists=True,
    )

    n_jobs = spec.n_jobs
    n_trials = spec.n_trials

    if n_jobs > 1:
        # Distribute trials across workers.  Give the first *remainder* workers
        # one extra trial each so we still reach exactly ``n_trials`` total.
        trials_per_worker = n_trials // n_jobs
        remainder = n_trials % n_jobs
        worker_trials = [
            trials_per_worker + (1 if i < remainder else 0)
            for i in range(n_jobs)
        ]
        print(f"[HPO] Running {n_trials} trials across {n_jobs} workers")
        print(f"[HPO] Trials per worker: {worker_trials}")

        ctx = multiprocessing.get_context("spawn")
        processes = []
        for worker_idx, worker_n in enumerate(worker_trials):
            if worker_n == 0:
                continue
            p = ctx.Process(
                target=_run_hpo_worker,
                args=(
                    str(config_path),
                    worker_n,
                    storage,
                    spec.study_name,
                    spec.direction,
                    spec.sampler_seed,
                    spec.sampler,
                    spec.n_startup_trials,
                    str(hpo_dir),
                    str(trials_dir),
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
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
        study.optimize(lambda trial: run_trial(trial, trial_context), n_trials=n_trials)

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
