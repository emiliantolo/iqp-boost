import json
from pathlib import Path

import numpy as np
import pytest

from src.datasets import DatasetBundle
from src.hpo.objective import resolve_objective_spec, validate_objective_before_training
from src.hpo.study import run_hpo
from src.run import run_boosting_experiment


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
    monkeypatch.setattr("src.hpo.finalization.evaluate_best_model", lambda *a, **k: None)

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
        lambda dataset_spec, config, plot_spec: DatasetBundle(
            dataset_name="Hamming Balls",
            x_train=np.zeros((4, 4), dtype=np.int8),
            x_test=np.ones((4, 4), dtype=np.int8),
        ),
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
    monkeypatch.setattr("src.hpo.finalization.evaluate_best_model", lambda *a, **k: None)
    config_path = tmp_path / "hpo.json"
    config_path.write_text(json.dumps(_config(tmp_path, objective_metric="test_mmd")))

    assert run_hpo(config_path).best_value == 0.33


def test_runner_errors_when_required_exact_sampling_would_fallback(monkeypatch, tmp_path):
    class FakeCircuit:
        n_qubits = 21
        bitflip = False

    monkeypatch.setattr("src.run.runner.setup_iqp_circuit", lambda *a, **k: (FakeCircuit(), [], "", None))
    monkeypatch.setattr("src.run.runner.save_circuit_plot", lambda *a, **k: None)
    monkeypatch.setattr("src.run.runner.compute_sigma", lambda *a, **k: 1.0)

    bundle = DatasetBundle(
        dataset_name="fake",
        x_train=np.zeros((2, 21), dtype=np.int8),
    )
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
            dataset=bundle,
            dataset_spec=None,
            output_base_dir=str(tmp_path),
            run_name="strict_exact",
        )
