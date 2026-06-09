import json
from types import SimpleNamespace

import numpy as np

from src.hpo.finalization import finalize_study, write_no_completed_trials_summary


def _study(best_trial=None):
    return SimpleNamespace(study_name="unit_hpo", best_trial=best_trial)


def _best_trial(run_dir, model_path, *, value=0.25):
    return SimpleNamespace(
        number=7,
        value=value,
        params={"learning_rate": 0.01},
        user_attrs={
            "output_dir": str(run_dir),
            "model_path": str(model_path),
            "final_stats": {"mmd": value},
            "ensemble_fcfw_weights": np.array([0.5, 0.5]),
        },
    )


def test_finalize_study_writes_best_artifacts_metrics_and_retrains(monkeypatch, tmp_path):
    run_dir = tmp_path / "trial_0007"
    run_dir.mkdir()
    model_path = run_dir / "ensemble.json"
    model_path.write_text(json.dumps({"models": [], "weights": []}))
    (run_dir / "config.json").write_text(json.dumps({"rng_seed": 7}))

    monkeypatch.setattr(
        "src.hpo.finalization.evaluate_best_model",
        lambda *args, **kwargs: ("hamming_balls_metrics", {"ensemble": {"recall": 1.0}}),
    )
    monkeypatch.setattr(
        "src.hpo.finalization.resolve_best_retrain_spec",
        lambda *args, **kwargs: SimpleNamespace(output_subdir="best_retrains"),
    )
    monkeypatch.setattr(
        "src.hpo.finalization.run_best_retrains",
        lambda *args, **kwargs: {"aggregates": {"final_stats": {"mmd": {"mean": 0.2}}}},
    )
    monkeypatch.setattr(
        "src.hpo.finalization.plot_best_retrain_summary",
        lambda *args, **kwargs: {"metrics_comparison_pdf": "plots/metrics_comparison.pdf"},
    )

    hpo_spec = {"dataset": {"name": "hamming_balls"}}
    summary = finalize_study(
        study=_study(_best_trial(run_dir, model_path)),
        hpo_spec=hpo_spec,
        config_path=tmp_path / "hpo.json",
        hpo_dir=tmp_path,
        objective_metric="mmd",
    )

    assert (tmp_path / "best_model.json").exists()
    assert json.loads((tmp_path / "best_config.json").read_text()) == {"rng_seed": 7}
    assert json.loads((tmp_path / "hpo_config.json").read_text()) == hpo_spec
    assert (tmp_path / "best_hamming_balls_metrics.json").exists()
    assert summary["best_trial"] == 7
    assert summary["best_config_path"] == str(tmp_path / "best_config.json")
    assert summary["hamming_balls_metrics"] == {"ensemble": {"recall": 1.0}}
    assert summary["best_retrains"]["aggregates"]["final_stats"]["mmd"]["mean"] == 0.2
    assert summary["best_retrains"]["plots"] == {
        "metrics_comparison_pdf": "plots/metrics_comparison.pdf"
    }
    assert json.loads((tmp_path / "study_summary.json").read_text())["best_value"] == 0.25


def test_finalize_study_without_best_config_uses_null_config_path(monkeypatch, tmp_path):
    run_dir = tmp_path / "trial_0007"
    run_dir.mkdir()
    model_path = run_dir / "ensemble.json"
    model_path.write_text(json.dumps({"models": [], "weights": []}))
    calls = {}

    def fake_evaluate_best_model(hpo_spec, best_config_path, best_model_path, fcfw_weights_list):
        calls["best_config_path"] = best_config_path
        calls["best_model_path"] = best_model_path
        return None

    def fake_run_best_retrains(spec, best_config_path, hpo_dir):
        calls["retrain_best_config_path"] = best_config_path
        return None

    monkeypatch.setattr("src.hpo.finalization.evaluate_best_model", fake_evaluate_best_model)
    monkeypatch.setattr(
        "src.hpo.finalization.resolve_best_retrain_spec",
        lambda *args, **kwargs: SimpleNamespace(output_subdir="best_retrains"),
    )
    monkeypatch.setattr("src.hpo.finalization.run_best_retrains", fake_run_best_retrains)
    monkeypatch.setattr("src.hpo.finalization.plot_best_retrain_summary", lambda *args, **kwargs: {})

    summary = finalize_study(
        study=_study(_best_trial(run_dir, model_path)),
        hpo_spec={"dataset": {"name": "hamming_balls"}},
        config_path=tmp_path / "hpo.json",
        hpo_dir=tmp_path,
        objective_metric="mmd",
    )

    assert (tmp_path / "best_model.json").exists()
    assert not (tmp_path / "best_config.json").exists()
    assert summary["best_config_path"] is None
    assert "hamming_balls_metrics" not in summary
    assert "best_retrains" not in summary
    assert calls["best_config_path"] is None
    assert calls["retrain_best_config_path"] is None


def test_finalize_study_uses_mmd_only_retrain_plots_for_mnist(monkeypatch, tmp_path):
    run_dir = tmp_path / "trial_0007"
    run_dir.mkdir()
    model_path = run_dir / "ensemble.json"
    model_path.write_text(json.dumps({"models": [], "weights": []}))
    (run_dir / "config.json").write_text(json.dumps({"rng_seed": 7}))
    calls = {}

    monkeypatch.setattr("src.hpo.finalization.evaluate_best_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "src.hpo.finalization.resolve_best_retrain_spec",
        lambda *args, **kwargs: SimpleNamespace(output_subdir="best_retrains"),
    )
    monkeypatch.setattr(
        "src.hpo.finalization.run_best_retrains",
        lambda *args, **kwargs: {"aggregates": {"final_stats": {"mmd": {"mean": 0.2}}}},
    )

    def fake_plot_best_retrain_summary(summary, hpo_dir, metric_filter=None, include_weight_distribution=True):
        calls["metric_filter"] = metric_filter
        calls["include_weight_distribution"] = include_weight_distribution
        return {"metrics_comparison_pdf": "plots/metrics_comparison.pdf"}

    monkeypatch.setattr("src.hpo.finalization.plot_best_retrain_summary", fake_plot_best_retrain_summary)

    finalize_study(
        study=_study(_best_trial(run_dir, model_path)),
        hpo_spec={"dataset": {"name": "mnist"}},
        config_path=tmp_path / "hpo.json",
        hpo_dir=tmp_path,
        objective_metric="test_mmd",
    )

    assert calls == {"metric_filter": "mmd", "include_weight_distribution": False}


def test_write_no_completed_trials_summary_writes_null_best_artifacts(tmp_path):
    hpo_spec = {"study_name": "unit_hpo"}
    summary = write_no_completed_trials_summary(
        study=_study(),
        hpo_spec=hpo_spec,
        config_path=tmp_path / "hpo.json",
        hpo_dir=tmp_path,
        objective_metric="mmd",
    )

    assert summary == {
        "study_name": "unit_hpo",
        "best_trial": None,
        "objective_metric": "mmd",
        "best_value": None,
        "best_params": {},
        "best_user_attrs": {},
        "best_model_path": None,
        "best_config_path": None,
        "config_path": str(tmp_path / "hpo.json"),
        "message": "No completed trials; all trials failed or stopped before completion.",
    }
    assert json.loads((tmp_path / "study_summary.json").read_text()) == summary
    assert json.loads((tmp_path / "hpo_config.json").read_text()) == hpo_spec
