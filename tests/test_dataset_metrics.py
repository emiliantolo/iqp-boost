from types import SimpleNamespace

import numpy as np

from src.datasets.dataset_metrics import (
    compute_and_save_dataset_metrics,
    compute_hamming_balls_metrics,
    compute_hopfield_metrics,
)


class RecordingOutput:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.saved = None

    def save_metrics(self, metrics, filename="metrics.json"):
        self.saved = {"metrics": metrics, "filename": filename}


def test_hopfield_metrics_are_finite_for_small_samples():
    patterns = np.array([[1.0, 1.0], [-1.0, -1.0]])
    J = np.array([[0.0, 1.0], [1.0, 0.0]])
    samples = np.array([[0, 0], [1, 1], [0, 1]], dtype=np.int8)

    metrics = compute_hopfield_metrics(samples, J, patterns)

    assert np.isfinite(metrics["energy_mean"])
    assert np.isfinite(metrics["energy_std"])
    assert np.isfinite(metrics["pattern_proximity"])
    assert 0.0 <= metrics["pattern_coverage"] <= 1.0


def test_hamming_balls_metrics_use_nearest_center_distance_and_threshold_coverage():
    centers = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int8)
    samples = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.int8)

    metrics = compute_hamming_balls_metrics(centers, samples, analytical_threshold=0.5)

    assert np.isclose(metrics["recall_distance"], 1.0 / 3.0)
    assert np.isclose(metrics["coverage"], 2.0 / 3.0)


def test_local_dataset_metrics_save_payload_for_hamming_balls(tmp_path, monkeypatch):
    monkeypatch.setattr("src.datasets.dataset_metrics.plt", None)
    dataset_obj = SimpleNamespace(
        centers=np.array([[0, 0], [1, 1]], dtype=np.int8),
        p=0.1,
        n_qubits=2,
        probs=np.array([0.45, 0.05, 0.05, 0.45]),
    )
    bundle = SimpleNamespace(
        dataset_name="Hamming Balls",
        x_train=np.array([[0, 0], [1, 1]], dtype=np.int8),
        dataset_obj=dataset_obj,
    )
    output = RecordingOutput(tmp_path)

    payload = compute_and_save_dataset_metrics(
        dataset_bundle=bundle,
        output=output,
        final_samples=np.array([[0, 0], [0, 1]], dtype=np.int8),
    )

    assert output.saved["filename"] == "dataset_metrics.json"
    assert payload["dataset"] == "hamming_balls"
    assert payload["final_ensemble"]["recall_distance"] == 0.5
