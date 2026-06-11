import json
from pathlib import Path

import numpy as np
import optuna
import pytest

from src.hpo.study import _make_sampler_from_settings, _worker_sampler_seed, run_hpo


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


def test_parallel_worker_samplers_do_not_share_startup_suggestions(tmp_path):
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(tmp_path / "journal.log"))
    )
    optuna.create_study(
        study_name="parallel_seed_hpo",
        storage=storage,
        sampler=_make_sampler_from_settings("tpe", 123, 16),
        load_if_exists=True,
    )

    suggestions = []
    for worker_idx in range(4):
        sampler = _make_sampler_from_settings(
            "tpe",
            _worker_sampler_seed(123, worker_idx),
            16,
        )
        study = optuna.create_study(
            study_name="parallel_seed_hpo",
            storage=storage,
            sampler=sampler,
            load_if_exists=True,
        )
        trial = study.ask()
        suggestions.append(
            (
                trial.suggest_float("boosting.learning_rate", 0.01, 1.0),
                trial.suggest_int("circuit_config.depth", 1, 10),
            )
        )

    assert len(set(suggestions)) == len(suggestions)


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
    hpo_dir = tmp_path / "hpo" / "unit_hpo"

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


def test_run_hpo_raises_when_parallel_worker_fails(monkeypatch, tmp_path):
    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.exitcode = 1

        def start(self):
            pass

        def join(self):
            pass

    class FakeContext:
        Process = FakeProcess

    monkeypatch.setattr("src.hpo.study.multiprocessing.get_context", lambda name: FakeContext())

    config = _base_hpo_config(tmp_path)
    config["n_jobs"] = 2
    config["n_trials"] = 2
    config_path = tmp_path / "hpo.json"
    config_path.write_text(json.dumps(config))

    with pytest.raises(RuntimeError, match="HPO worker\\(s\\) failed"):
        run_hpo(config_path)


def test_run_hpo_skips_when_target_completed_trials_already_exist(monkeypatch, tmp_path):
    calls = []

    def fake_run_boosting_experiment(**kwargs):
        calls.append(kwargs)
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

    monkeypatch.setattr("src.hpo.trial.build_dataset_bundle", lambda **kwargs: {
        "dataset_name": "Hamming Balls",
        "x_train": np.zeros((4, 4), dtype=np.int8),
        "validity_fn": None,
        "coverage_fn": None,
        "custom_viz_fn": None,
    })
    monkeypatch.setattr("src.hpo.trial.run_boosting_experiment", fake_run_boosting_experiment)
    monkeypatch.setattr("src.hpo.finalization.evaluate_best_model", lambda *args, **kwargs: None)

    config_path = tmp_path / "hpo.json"
    config_path.write_text(json.dumps(_base_hpo_config(tmp_path)))

    run_hpo(config_path)
    hpo_dir = tmp_path / "hpo" / "unit_hpo"
    assert (hpo_dir / "study_summary.json").exists()

    run_hpo(config_path)

    assert len(calls) == 1


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
