import json
from pathlib import Path

import numpy as np
import pytest

from src.hpo.study import run_hpo


class FakeEnsemble:
    def save(self, path):
        Path(path).write_text(json.dumps({"models": [], "weights": []}))


def _base_hpo_config(tmp_path, objective_metric="mmd"):
    return {
        "study_name": "unit_hpo",
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
        "objective_metric": objective_metric,
    }


def test_run_hpo_writes_summary_and_hamming_balls_metrics(monkeypatch, tmp_path):
    def fake_bundle(dataset_spec, config, plot_spec):
        return {
            "dataset_name": "Hamming Balls",
            "x_train": np.zeros((4, 4), dtype=np.int8),
            "validity_fn": None,
            "coverage_fn": None,
            "custom_viz_fn": None,
            "top_k_tvd_fn": None,
            "exact_probs": None,
            "generation_eval_fn": None,
        }

    def fake_run_boosting_experiment(**kwargs):
        output_dir = Path(kwargs["output_base_dir"]) / kwargs["run_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(kwargs["config"]))
        return {
            "final_stats": {"mmd": 0.25},
            "ensemble": FakeEnsemble(),
            "n_models_accepted": 1,
            "weights": np.array([1.0]),
            "ensemble_fcfw_weights": None,
            "output_dir": str(output_dir),
        }

    class FakeLoadedEnsemble:
        def sample(self, shots, rng, weights_override=None):
            return np.zeros((shots, 4), dtype=np.int8)

    monkeypatch.setattr("src.hpo.trial.build_dataset_bundle", fake_bundle)
    monkeypatch.setattr("src.hpo.trial.run_boosting_experiment", fake_run_boosting_experiment)
    monkeypatch.setattr("src.hpo.best_model_evaluation.setup_iqp_circuit", lambda *a, **k: (object(), [], "", None))
    monkeypatch.setattr("src.hpo.best_model_evaluation.BoostedEnsemble.load", lambda *a, **k: FakeLoadedEnsemble())

    config_path = tmp_path / "hpo.json"
    config_path.write_text(json.dumps(_base_hpo_config(tmp_path)))

    study = run_hpo(config_path)
    hpo_dir = next((tmp_path / "hpo").glob("unit_hpo_*"))

    assert study.best_value == 0.25
    assert (hpo_dir / "study_summary.json").exists()
    assert (hpo_dir / "hpo_config.json").exists()
    assert (hpo_dir / "best_model.npz").exists()
    assert (hpo_dir / "best_config.json").exists()
    assert (hpo_dir / "best_hamming_balls_metrics.json").exists()

    summary = json.loads((hpo_dir / "study_summary.json").read_text())
    assert summary["objective_metric"] == "mmd"
    assert summary["best_value"] == 0.25
    assert "hamming_balls_metrics" in summary


def test_run_hpo_rejects_invalid_objective_metric(monkeypatch, tmp_path):
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

    def fake_run_boosting_experiment(**kwargs):
        output_dir = Path(kwargs["output_base_dir"]) / kwargs["run_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(kwargs["config"]))
        return {
            "final_stats": {"mmd": 0.25},
            "ensemble": FakeEnsemble(),
            "n_models_accepted": 1,
            "weights": np.array([1.0]),
            "output_dir": str(output_dir),
        }

    monkeypatch.setattr("src.hpo.trial.run_boosting_experiment", fake_run_boosting_experiment)
    config = _base_hpo_config(tmp_path, objective_metric="missing_metric")
    config_path = tmp_path / "hpo.json"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="produced invalid missing_metric"):
        run_hpo(config_path)
