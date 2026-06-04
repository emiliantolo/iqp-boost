import json
from pathlib import Path

import numpy as np
import optuna
import pytest

from src.hpo.objective import resolve_objective_spec, validate_objective_before_training
from src.hpo.pruning import build_pruning_spec, make_trial_pruning_callback
from src.hpo.study import run_hpo
from src.runner import run_boosting_experiment


class FakeEnsemble:
    def save(self, path):
        Path(path).write_text(json.dumps({"models": [], "weights": []}))


def _config(tmp_path, **overrides):
    config = {
        "study_name": "objective_hpo",
        "n_trials": 1,
        "sampler_seed": 123,
        "output_dir": str(tmp_path / "hpo"),
        "dataset": {"name": "hamming_balls", "params": {"n_qubits": 4, "K": 2, "p": 0.1}},
        "plot": {"kind": "none"},
        "fixed_config": {
            "rng_seed": 7,
            "shots": 8,
            "n_samples": 4,
            "circuit_config": {"topology": "neighbour", "distance": 1, "max_weight": 1},
        },
        "search_space": {},
        "objective_metric": "mmd",
    }
    config.update(overrides)
    return config


def test_exact_tvd_objective_maps_to_final_tvd(monkeypatch, tmp_path):
    def fake_bundle(dataset_spec, config, plot_spec):
        return {
            "dataset_name": "Hamming Balls",
            "x_train": np.zeros((4, 4), dtype=np.int8),
            "validity_fn": None,
            "coverage_fn": None,
            "custom_viz_fn": None,
        }

    def fake_run_boosting_experiment(**kwargs):
        assert kwargs["config"]["require_exact_sampling"] is True
        output_dir = Path(kwargs["output_base_dir"]) / kwargs["run_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(kwargs["config"]))
        return {
            "final_stats": {"tvd": 0.125},
            "ensemble": FakeEnsemble(),
            "n_models_accepted": 1,
            "weights": np.array([1.0]),
            "output_dir": str(output_dir),
        }

    monkeypatch.setattr("src.hpo.trial.build_dataset_bundle", fake_bundle)
    monkeypatch.setattr("src.hpo.trial.run_boosting_experiment", fake_run_boosting_experiment)
    monkeypatch.setattr("src.hpo.study.evaluate_best_model", lambda *a, **k: None)

    config = _config(tmp_path, objective_metric="exact_tvd")
    config["fixed_config"]["exact_sampling"] = True
    config["fixed_config"]["skip_sampling"] = True
    config["fixed_config"]["final_eval_sampling"] = True
    config_path = tmp_path / "hpo.json"
    config_path.write_text(json.dumps(config))

    study = run_hpo(config_path)
    hpo_dir = next((tmp_path / "hpo").glob("objective_hpo_*"))
    summary = json.loads((hpo_dir / "study_summary.json").read_text())
    assert study.best_value == 0.125
    assert summary["objective_metric"] == "exact_tvd"
    assert summary["best_value"] == 0.125


def test_exact_tvd_requires_exact_and_final_sampling_flags():
    objective = resolve_objective_spec({"objective_metric": "exact_tvd"})
    with pytest.raises(ValueError, match="exact_sampling=true"):
        validate_objective_before_training(objective, {"exact_sampling": False}, {})

    with pytest.raises(ValueError, match="requires final sampling"):
        validate_objective_before_training(
            objective,
            {"exact_sampling": True, "skip_sampling": True, "final_eval_sampling": False},
            {},
        )


def test_test_mmd_requires_x_test_before_training():
    objective = resolve_objective_spec({"objective_metric": "test_mmd"})
    with pytest.raises(ValueError, match="requires the dataset bundle to provide x_test"):
        validate_objective_before_training(objective, {}, {"x_train": np.zeros((2, 2))})


def test_test_mmd_objective_uses_final_stats(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.hpo.trial.build_dataset_bundle",
        lambda dataset_spec, config, plot_spec: {
            "dataset_name": "Hamming Balls",
            "x_train": np.zeros((4, 4), dtype=np.int8),
            "x_test": np.ones((4, 4), dtype=np.int8),
            "validity_fn": None,
            "coverage_fn": None,
            "custom_viz_fn": None,
        },
    )

    def fake_run_boosting_experiment(**kwargs):
        output_dir = Path(kwargs["output_base_dir"]) / kwargs["run_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(kwargs["config"]))
        return {
            "final_stats": {"test_mmd": 0.33},
            "ensemble": FakeEnsemble(),
            "n_models_accepted": 1,
            "weights": np.array([1.0]),
            "output_dir": str(output_dir),
        }

    monkeypatch.setattr("src.hpo.trial.run_boosting_experiment", fake_run_boosting_experiment)
    monkeypatch.setattr("src.hpo.study.evaluate_best_model", lambda *a, **k: None)
    config_path = tmp_path / "hpo.json"
    config_path.write_text(json.dumps(_config(tmp_path, objective_metric="test_mmd")))

    assert run_hpo(config_path).best_value == 0.33


def test_hyperband_defaults_resolve_n_models():
    pruning = build_pruning_spec(
        {"pruner": {"type": "hyperband", "metric": "training_mmd"}},
        {"n_models": 7},
    )
    assert pruning.enabled
    assert pruning.metric == "training_mmd"
    assert isinstance(pruning.pruner, optuna.pruners.HyperbandPruner)
    assert pruning.pruner._min_resource == 1
    assert pruning.pruner._max_resource == 7
    assert pruning.pruner._reduction_factor == 3


def test_hyperband_max_resource_uses_n_models_search_upper_bound():
    pruning = build_pruning_spec(
        {
            "search_space": {"n_models": {"type": "int", "low": 2, "high": 9}},
            "pruner": {"type": "hyperband", "metric": "training_mmd"},
        },
        {"n_models": 4},
    )
    assert pruning.pruner._max_resource == 9


def test_pruning_callback_reports_and_prunes():
    class FakeTrial:
        def __init__(self):
            self.reports = []

        def report(self, value, step):
            self.reports.append((value, step))

        def should_prune(self):
            return True

    pruning = build_pruning_spec(
        {"pruner": {"type": "median", "metric": "training_mmd"}},
        {"n_models": 3},
    )
    trial = FakeTrial()
    callback = make_trial_pruning_callback(
        trial,
        pruning,
        resolve_objective_spec({"objective_metric": "mmd"}),
    )

    with pytest.raises(optuna.TrialPruned):
        callback({"step": 2, "training_mmd": 0.4})
    assert trial.reports == [(0.4, 2)]


def test_objective_metric_pruning_skips_unavailable_intermediate_metric():
    class FakeTrial:
        def __init__(self):
            self.reports = []

        def report(self, value, step):
            self.reports.append((value, step))

        def should_prune(self):
            return True

    pruning = build_pruning_spec(
        {"pruner": {"type": "hyperband", "metric": "objective_metric"}},
        {"n_models": 3},
    )
    trial = FakeTrial()
    callback = make_trial_pruning_callback(
        trial,
        pruning,
        resolve_objective_spec({"objective_metric": "exact_tvd"}),
    )
    callback({"step": 1, "training_mmd": 0.4, "tvd": 0.2})
    assert trial.reports == []


def test_all_pruned_trials_skip_best_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.hpo.trial.build_dataset_bundle",
        lambda dataset_spec, config, plot_spec: {
            "dataset_name": "Hamming Balls",
            "x_train": np.zeros((4, 4), dtype=np.int8),
            "validity_fn": None,
            "coverage_fn": None,
            "custom_viz_fn": None,
        },
    )

    def fake_pruning_callback(trial, pruning, objective):
        def _callback(payload):
            raise optuna.TrialPruned("forced prune")

        return _callback

    def fake_run_boosting_experiment(**kwargs):
        kwargs["hpo_callback"]({"step": 0, "training_mmd": 1.0})
        raise AssertionError("callback should prune before this point")

    monkeypatch.setattr("src.hpo.trial.make_trial_pruning_callback", fake_pruning_callback)
    monkeypatch.setattr("src.hpo.trial.run_boosting_experiment", fake_run_boosting_experiment)
    config = _config(
        tmp_path,
        pruner={"type": "median", "metric": "training_mmd", "n_startup_trials": 0, "n_warmup_steps": 0},
    )
    config_path = tmp_path / "hpo.json"
    config_path.write_text(json.dumps(config))

    study = run_hpo(config_path)
    hpo_dir = next((tmp_path / "hpo").glob("objective_hpo_*"))
    summary = json.loads((hpo_dir / "study_summary.json").read_text())
    assert summary["best_trial"] is None
    assert not (hpo_dir / "best_model.json").exists()
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED


def test_runner_errors_when_required_exact_sampling_would_fallback(monkeypatch, tmp_path):
    class FakeCircuit:
        n_qubits = 21
        bitflip = False

    monkeypatch.setattr("src.runner.setup_iqp_circuit", lambda *a, **k: (FakeCircuit(), [], "", None))
    monkeypatch.setattr("src.runner.save_circuit_plot", lambda *a, **k: None)
    monkeypatch.setattr("src.runner.compute_sigma", lambda *a, **k: 1.0)
    config = {
        "rng_seed": 1,
        "n_models": 1,
        "n_samples": 2,
        "shots": 2,
        "n_ops": 2,
        "exact_sampling": True,
        "require_exact_sampling": True,
        "circuit_config": {},
    }

    with pytest.raises(ValueError, match="n_qubits > 20"):
        run_boosting_experiment(
            config=config,
            dataset_name="fake",
            dataset_spec=None,
            x_train=np.zeros((2, 21), dtype=np.int8),
            validity_fn=None,
            coverage_fn=None,
            output_base_dir=str(tmp_path),
            run_name="strict_exact",
        )
