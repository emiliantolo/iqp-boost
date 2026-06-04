import math

import numpy as np

from src.benchmark_metrics import compute_hamming_balls_metrics


def test_hamming_balls_exact_centers_have_zero_recall_distance():
    centers = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int8)
    metrics = compute_hamming_balls_metrics(centers, centers, n_qubits=3)
    assert metrics["recall_distance"] == 0.0


def test_hamming_balls_coverage_is_full_when_all_centers_hit():
    centers = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int8)
    samples = np.array([[0, 0, 0, 1], [1, 1, 1, 0]], dtype=np.int8)
    metrics = compute_hamming_balls_metrics(centers, samples, n_qubits=4, radius_fraction=0.25)
    assert metrics["coverage"] == 1.0
    assert metrics["per_center_coverage"] == [True, True]
    assert metrics["per_center_hit_counts"] == [1, 1]
    assert metrics["mean_distance_to_nearest_center"] == metrics["recall_distance"]
    assert metrics["median_distance_to_nearest_center"] == 1.0


def test_hamming_balls_coverage_can_be_partial():
    centers = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int8)
    samples = np.array([[0, 0, 0, 1]], dtype=np.int8)
    metrics = compute_hamming_balls_metrics(centers, samples, n_qubits=4, radius_fraction=0.25)
    assert metrics["coverage"] == 0.5


def test_hamming_balls_empty_samples_are_stable():
    centers = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int8)
    metrics = compute_hamming_balls_metrics(centers, np.empty((0, 3), dtype=np.int8), n_qubits=3)
    assert math.isnan(metrics["recall_distance"])
    assert metrics["coverage"] == 0.0
    assert metrics["per_center_coverage"] == [False, False]
    assert metrics["per_center_hit_counts"] == [0, 0]
    assert math.isnan(metrics["mean_distance_to_nearest_center"])
    assert math.isnan(metrics["median_distance_to_nearest_center"])
