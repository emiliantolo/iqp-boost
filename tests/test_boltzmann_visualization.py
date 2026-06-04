import numpy as np

import src.datasets.boltzmann_visualization as viz
from src.datasets.boltzmann_metrics import spin_covariance_from_probs, spin_covariance_from_samples


class DummyFigure:
    def savefig(self, *args, **kwargs):
        pass


class DummyOutput:
    def get_path(self, filename):
        return f"/tmp/{filename}"


def test_covariance_visualization_uses_samples_for_baseline_and_ensemble(monkeypatch):
    captured = {}
    exact_probs = np.array([0.5, 0.0, 0.0, 0.5])
    x_train = np.array([[0, 0], [1, 1]], dtype=np.int8)
    baseline_samples = np.array([[0, 0], [0, 0]], dtype=np.int8)
    ensemble_samples = np.array([[0, 1], [0, 1]], dtype=np.int8)
    dataset = type("Dataset", (), {"probs": exact_probs})()

    monkeypatch.setattr(viz, "plot_hamming_weight_histogram", lambda *args, **kwargs: DummyFigure())
    monkeypatch.setattr(viz, "plot_lorenz_curve", lambda *args, **kwargs: DummyFigure())

    def capture_covariance(reference, baseline, ensemble):
        captured["reference"] = reference
        captured["baseline"] = baseline
        captured["ensemble"] = ensemble
        return DummyFigure()

    monkeypatch.setattr(viz, "plot_covariance_heatmaps", capture_covariance)

    viz.generate_boltzmann_visualizations(
        DummyOutput(),
        x_train,
        baseline_samples,
        ensemble_samples,
        per_model_samples=[],
        weights=np.array([1.0]),
        dataset=dataset,
    )

    assert np.array_equal(captured["reference"], spin_covariance_from_probs(exact_probs))
    assert np.array_equal(captured["baseline"], spin_covariance_from_samples(baseline_samples))
    assert np.array_equal(captured["ensemble"], spin_covariance_from_samples(ensemble_samples))
    assert not np.array_equal(captured["baseline"], captured["reference"])
    assert not np.array_equal(captured["ensemble"], captured["reference"])
