import json
from pathlib import Path

import numpy as np

from src.datasets import DatasetBundle
from src.hpo.objective import resolve_objective_spec
from src.hpo.trial import HpoTrialContext, run_trial


class FakeEnsemble:
    def save(self, path):
        Path(path).write_text(json.dumps({"models": [], "weights": []}))


class FakeTrial:
    number = 3

    def __init__(self):
        self.user_attrs = {}
        self.reports = []

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value

    def report(self, value, step):
        self.reports.append((value, step))

    def should_prune(self):
        return False


def _context(tmp_path, objective_metric="mmd"):
    base_config = {
        "n_models": 4,
        "rng_seed": 7,
        "shots": 8,
        "n_samples": 4,
        "circuit_config": {"topology": "neighbour", "distance": 1, "max_weight": 1},
    }
    hpo_spec = {
        "objective_metric": objective_metric,
    }
    return HpoTrialContext(
        base_config=base_config,
        search_space={},
        dataset_spec={"name": "hamming_balls", "params": {"n_qubits": 4}},
        plot_spec={"kind": "none"},
        objective=resolve_objective_spec(hpo_spec),
        trials_dir=tmp_path / "trials",
        hpo_dir=tmp_path / "hpo",
        metric_configs=[("mmd", "MMD", 1, "blue", "o")],
        baseline_epochs=12,
    )


def test_run_trial_records_attrs_saves_model_and_forwards_x_test(monkeypatch, tmp_path):
    captured = {}

    def fake_bundle(dataset_spec, config, plot_spec):
        return DatasetBundle(
            dataset_name="Hamming Balls",
            x_train=np.zeros((4, 4), dtype=np.int8),
            x_test=np.ones((4, 4), dtype=np.int8),
        )

    def fake_run_boosting_experiment(**kwargs):
        captured.update(kwargs)
        output_dir = Path(kwargs["output_base_dir"]) / kwargs["run_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "final_stats": {"test_mmd": 0.42},
            "ensemble": FakeEnsemble(),
            "n_models_accepted": 2,
            "weights": np.array([0.25, 0.75]),
            "ensemble_fcfw_weights": np.array([0.5, 0.5]),
            "output_dir": str(output_dir),
        }

    monkeypatch.setattr("src.hpo.trial.build_dataset_bundle", fake_bundle)
    monkeypatch.setattr("src.hpo.trial.run_boosting_experiment", fake_run_boosting_experiment)

    trial = FakeTrial()
    value = run_trial(trial, _context(tmp_path, objective_metric="test_mmd"))

    assert value == 0.42
    assert captured["dataset"].x_test.shape == (4, 4)
    assert captured["baseline_epochs"] == 12
    assert captured["metric_configs"][0][0] == "mmd"
    assert Path(trial.user_attrs["model_path"]).exists()
    assert trial.user_attrs["final_stats"] == {"test_mmd": 0.42}
    assert trial.user_attrs["n_models_accepted"] == 2
    assert trial.user_attrs["weights"] == [0.25, 0.75]
    assert trial.user_attrs["ensemble_fcfw_weights"] == [0.5, 0.5]
    assert trial.reports == []


def test_run_trial_tvd_exact_sets_require_exact_sampling(monkeypatch, tmp_path):
    captured = {}

    def fake_bundle(dataset_spec, config, plot_spec):
        return {
            "dataset_name": "Hamming Balls",
            "x_train": np.zeros((4, 4), dtype=np.int8),
            "validity_fn": None,
            "coverage_fn": None,
            "custom_viz_fn": None,
        }

    def fake_run_boosting_experiment(**kwargs):
        captured.update(kwargs)
        output_dir = Path(kwargs["output_base_dir"]) / kwargs["run_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "final_stats": {"tvd": 0.9, "tvd_exact": 0.2},
            "ensemble": FakeEnsemble(),
            "n_models_accepted": 1,
            "weights": np.array([1.0]),
            "output_dir": str(output_dir),
        }

    monkeypatch.setattr("src.hpo.trial.build_dataset_bundle", fake_bundle)
    monkeypatch.setattr("src.hpo.trial.run_boosting_experiment", fake_run_boosting_experiment)

    context = _context(tmp_path, objective_metric="tvd_exact")
    context.base_config["exact_sampling"] = True
    context.base_config["skip_sampling"] = True
    context.base_config["final_eval_sampling"] = True

    assert run_trial(FakeTrial(), context) == 0.2
    assert captured["config"]["require_exact_sampling"] is True
