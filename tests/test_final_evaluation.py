import numpy as np

from src.dataset_catalog import DatasetBundle
from src.final_evaluation import FcfwEvaluationResult, FinalEvaluationContext, run_final_evaluation


class RecordingOutput:
    def __init__(self):
        self.saved = None

    def save_results_csv(self, metrics_history, baseline_stats=None, summary_rows=None):
        self.saved = {
            "metrics_history": metrics_history,
            "baseline_stats": baseline_stats,
            "summary_rows": summary_rows,
        }


class FakeEvaluation:
    def __init__(self, final_sampling_enabled=False, sampling_enabled=False):
        self.final_sampling_enabled = final_sampling_enabled
        self.sampling_enabled = sampling_enabled
        self.sample_evaluations = []
        self.training_evaluations = []

    def evaluate_samples(self, samples):
        self.sample_evaluations.append(np.asarray(samples))
        return {"mmd": float(np.asarray(samples).sum())}

    def evaluate_ensemble_training_mmd(self, ensemble, ground_truth=None):
        self.training_evaluations.append(ground_truth)
        if ground_truth is None:
            return 0.5
        return float(np.asarray(ground_truth).sum())


class FakeEnsemble:
    def __init__(self):
        self.models = [np.array([0]), np.array([1])]
        self.weights = [0.5, 0.5]
        self.sample_calls = []

    def sample(self, shots, rng, return_details=False):
        self.sample_calls.append({"shots": shots, "return_details": return_details})
        final_samples = np.ones((shots, 2), dtype=np.int8)
        if not return_details:
            return final_samples
        per_model = [
            np.zeros((1, 2), dtype=np.int8),
            np.ones((1, 2), dtype=np.int8),
        ]
        return final_samples, np.array([1, 1]), per_model


def test_analytical_final_evaluation_writes_final_stats_and_no_samples(monkeypatch):
    _disable_plotting(monkeypatch)
    ensemble = FakeEnsemble()
    context = _context(
        ensemble=ensemble,
        standalone_stats={"mmd": 1.0},
        ensemble_metrics_history={"mmd": [0.8], "training_loss": [0.7, 0.6], "step": [0], "alpha": [1.0]},
        reference_label="Standalone",
        report_fcfw=False,
    )

    result = run_final_evaluation(context)

    assert result.final_stats == {"mmd": 0.6}
    assert result.final_ensemble_samples is None
    assert result.per_model_samples == []
    assert ensemble.sample_calls == []
    assert context.output.saved["summary_rows"][-1] == {
        "step": -4,
        "label": "ensemble_final",
        "metrics": result.final_stats,
    }


def test_sampled_final_evaluation_reuses_final_sample_for_per_model_rows(monkeypatch):
    _disable_plotting(monkeypatch)
    ensemble = FakeEnsemble()
    evaluation = FakeEvaluation(final_sampling_enabled=True, sampling_enabled=True)
    context = _context(
        ensemble=ensemble,
        evaluation=evaluation,
        shots=3,
        report_fcfw=False,
    )

    result = run_final_evaluation(context)

    assert len(ensemble.sample_calls) == 1
    assert ensemble.sample_calls[0] == {"shots": 3, "return_details": True}
    assert np.array_equal(result.final_ensemble_samples, np.ones((3, 2), dtype=np.int8))
    assert len(result.per_model_samples) == 2
    assert len(evaluation.sample_evaluations) == 3


def test_fcfw_disabled_omits_fcfw_rows_and_weights(monkeypatch):
    _disable_plotting(monkeypatch)
    context = _context(
        data_only_stats={"mmd": 0.9},
        reference_label="Standalone",
        report_fcfw=False,
    )

    result = run_final_evaluation(context)

    labels = [row["label"] for row in context.output.saved["summary_rows"]]
    assert "data_only_fcfw" not in labels
    assert "ensemble_fcfw" not in labels
    assert result.data_only_fcfw_weights is None
    assert result.ensemble_fcfw_weights is None


def test_test_mmd_is_saved_with_final_summary(monkeypatch):
    _disable_plotting(monkeypatch)
    dataset = DatasetBundle(
        "Synthetic",
        np.array([[0, 0]], dtype=np.int8),
        x_test=np.array([[1, 1]], dtype=np.int8),
    )
    context = _context(
        dataset=dataset,
        report_fcfw=False,
    )

    result = run_final_evaluation(context)

    final_summary = context.output.saved["summary_rows"][-1]
    assert result.final_stats["test_mmd"] == 2.0
    assert final_summary["metrics"]["test_mmd"] == 2.0


def test_fcfw_enabled_adds_summary_rows_and_weights(monkeypatch):
    _disable_plotting(monkeypatch)
    calls = []

    def fake_fcfw(*args, **kwargs):
        calls.append((args, kwargs))
        return FcfwEvaluationResult(
            metrics={"mmd": 0.123, "tvd": float("nan")},
            weights=np.array([1.0]),
        )

    monkeypatch.setattr("src.final_evaluation.compute_fcfw_stats", fake_fcfw)
    context = _context(
        data_only_ensemble=FakeEnsemble(),
        data_only_stats={"mmd": 0.9},
        reference_label="Standalone",
        report_fcfw=True,
    )

    result = run_final_evaluation(context)

    labels = [row["label"] for row in context.output.saved["summary_rows"]]
    assert labels == ["data_only", "data_only_fcfw", "ensemble_final", "ensemble_fcfw"]
    assert np.array_equal(result.data_only_fcfw_weights, np.array([1.0]))
    assert np.array_equal(result.ensemble_fcfw_weights, np.array([1.0]))
    assert len(calls) == 2


def test_data_only_fcfw_skipped_when_data_only_is_reference(monkeypatch):
    _disable_plotting(monkeypatch)
    calls = []

    def fake_fcfw(*args, **kwargs):
        calls.append((args, kwargs))
        return FcfwEvaluationResult(metrics={"mmd": 0.123}, weights=np.array([1.0]))

    monkeypatch.setattr("src.final_evaluation.compute_fcfw_stats", fake_fcfw)
    context = _context(
        data_only_ensemble=FakeEnsemble(),
        data_only_stats={"mmd": 0.9},
        reference_label="Data-only",
        report_fcfw=True,
    )

    result = run_final_evaluation(context)

    labels = [row["label"] for row in context.output.saved["summary_rows"]]
    assert "data_only" in labels
    assert "data_only_fcfw" not in labels
    assert result.data_only_fcfw_weights is None
    assert result.ensemble_fcfw_weights is not None
    assert len(calls) == 1


def test_final_evaluation_context_does_not_require_baseline_samples(monkeypatch):
    _disable_plotting(monkeypatch)
    context = _context(report_fcfw=False)

    result = run_final_evaluation(context)

    assert result.final_stats["mmd"] == 0.7


def _context(
    *,
    config=None,
    dataset=None,
    output=None,
    ensemble=None,
    data_only_ensemble=None,
    data_only_stats=None,
    data_only_history=None,
    standalone_stats=None,
    baseline_train_losses=None,
    ensemble_metrics_history=None,
    evaluation=None,
    sigma=1.0,
    shots=4,
    rng_seed=11,
    reference_stats=None,
    reference_label="Synthetic",
    metric_configs=None,
    report_fcfw=False,
) -> FinalEvaluationContext:
    config = {"n_models": 1, "report_fcfw": report_fcfw, **(config or {})}
    return FinalEvaluationContext(
        config=config,
        dataset=dataset or DatasetBundle("Synthetic", np.array([[0, 0]], dtype=np.int8)),
        output=output or RecordingOutput(),
        ensemble=ensemble or FakeEnsemble(),
        data_only_ensemble=data_only_ensemble,
        data_only_stats=data_only_stats,
        data_only_history=data_only_history,
        standalone_stats=standalone_stats,
        baseline_train_losses=baseline_train_losses,
        ensemble_metrics_history=ensemble_metrics_history or {
            "mmd": [0.8],
            "training_loss": [0.7],
            "step": [0],
            "alpha": [1.0],
        },
        evaluation=evaluation or FakeEvaluation(final_sampling_enabled=False),
        sigma=sigma,
        shots=shots,
        rng_seed=rng_seed,
        reference_stats=reference_stats or {"mmd": 1.0},
        reference_label=reference_label,
        metric_configs=metric_configs,
    )


def _disable_plotting(monkeypatch):
    monkeypatch.setattr("src.final_evaluation.get_plot_config", lambda: {"plot_data_loss": False})
    monkeypatch.setattr("src.final_evaluation.report_final", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.final_evaluation.report_metrics_table", lambda *args, **kwargs: None)
