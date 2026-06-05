import json
from pathlib import Path

import numpy as np
import pytest

from src.hpo.best_retrains import resolve_best_retrain_spec, run_best_retrains


class FakeEnsemble:
    def save(self, path):
        Path(path).write_text(json.dumps({"models": [], "weights": []}))


def test_run_best_retrains_writes_seed_artifacts_and_aggregates(monkeypatch, tmp_path):
    captured_configs = []

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
        captured_configs.append(dict(kwargs["config"]))
        output_dir = Path(kwargs["output_base_dir"]) / kwargs["run_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        idx = int(kwargs["run_name"].split("_")[-1])
        return {
            "final_stats": {
                "tvd": 0.1 + idx,
                "mmd": 0.2 + idx,
                "recall_distance": 1.0 + idx,
                "coverage": 0.5,
            },
            "baseline_stats": {"tvd": 0.3 + idx},
            "ensemble_fcfw_stats": {"tvd": 0.05 + idx},
            "ensemble": FakeEnsemble(),
            "n_models_accepted": 2,
            "weights": np.array([0.25, 0.75]),
            "ensemble_fcfw_weights": np.array([0.5, 0.5]),
            "output_dir": str(output_dir),
        }

    monkeypatch.setattr("src.hpo.best_retrains.build_dataset_bundle", fake_bundle)
    monkeypatch.setattr("src.hpo.best_retrains.run_boosting_experiment", fake_run_boosting_experiment)

    best_config_path = tmp_path / "best_config.json"
    best_config_path.write_text(
        json.dumps(
            {
                "rng_seed": 42,
                "data_seed": 42,
                "baseline": "none",
                "skip_sampling": True,
                "final_eval_sampling": True,
                "report_fcfw": False,
            }
        )
    )
    hpo_spec = {
        "dataset": {"name": "hamming_balls", "params": {"n_qubits": 4}},
        "plot": {"kind": "none"},
        "best_retrains": {"n_seeds": 5, "baseline": "standalone", "report_fcfw": True},
    }

    spec = resolve_best_retrain_spec(hpo_spec)
    summary = run_best_retrains(spec, best_config_path, tmp_path)

    assert len(captured_configs) == 5
    assert all(config["baseline"] == "standalone" for config in captured_configs)
    assert all(config["report_fcfw"] is True for config in captured_configs)
    assert captured_configs[0]["rng_seed"] == 0
    assert captured_configs[4]["data_seed"] == 4
    assert (tmp_path / "best_retrains" / "seed_000" / "ensemble.json").exists()
    assert (tmp_path / "best_retrains" / "seed_000" / "final_stats.json").exists()
    assert (tmp_path / "best_retrains" / "summary.json").exists()
    assert summary["aggregates"]["final_stats"]["tvd"]["mean"] == pytest.approx(2.1)
    assert summary["aggregates"]["baseline_stats"]["tvd"]["mean"] == pytest.approx(2.3)
    assert summary["aggregates"]["ensemble_fcfw_stats"]["tvd"]["mean"] == pytest.approx(2.05)


def test_resolve_best_retrain_spec_returns_none_without_config():
    assert resolve_best_retrain_spec({"dataset": {"name": "hamming_balls"}}) is None


def test_resolve_best_retrain_spec_applies_defaults():
    spec = resolve_best_retrain_spec(
        {
            "dataset": {"name": "hamming_balls", "params": {"n_qubits": 4}},
            "best_retrains": {},
        }
    )

    assert spec.dataset_spec == {"name": "hamming_balls", "params": {"n_qubits": 4}}
    assert spec.plot_spec == {"kind": "none"}
    assert spec.n_seeds == 5
    assert spec.seed_start == 0
    assert spec.baseline == "standalone"
    assert spec.report_fcfw is True
    assert spec.skip_sampling is True
    assert spec.final_eval_sampling is True
    assert spec.output_subdir == "best_retrains"


def test_run_best_retrains_skips_missing_best_config(tmp_path):
    spec = resolve_best_retrain_spec(
        {
            "dataset": {"name": "hamming_balls", "params": {"n_qubits": 4}},
            "best_retrains": {"n_seeds": 1},
        }
    )

    assert run_best_retrains(spec, tmp_path / "missing.json", tmp_path) is None
