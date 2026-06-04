import math

import numpy as np
import pytest

from src.core import (
    EvaluationPolicy,
    compute_ensemble_training_mmd,
    evaluate_samples,
    marginalize_probs_to_wires,
    reorder_probs_to_sample_indexing,
)
from src.core.metrics import compute_tvd


def test_sampled_evaluation_keeps_metric_keys_and_exact_probability_tvd():
    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    samples = np.array([[0, 0], [0, 0], [1, 0], [1, 1]], dtype=np.int8)
    exact_probs = np.full(4, 0.25)

    stats = evaluate_samples(x_train, samples, sigma=1.0, exact_probs=exact_probs)

    assert set(stats) == {
        "mmd",
        "kl",
        "jsd",
        "tvd",
        "validity",
        "coverage",
        "precision",
        "recall",
        "support_match",
        "f_score",
        "corr_fro",
    }
    assert np.isclose(stats["tvd"], 0.25)
    assert math.isfinite(stats["kl"])
    assert math.isfinite(stats["jsd"])


def test_compute_tvd_accepts_exact_model_probs_without_replacing_sampled_path():
    x_train = np.array([[0, 0], [1, 1]], dtype=np.int8)
    samples = np.array([[0, 0], [0, 0]], dtype=np.int8)
    exact_probs = np.array([0.5, 0.0, 0.0, 0.5])
    model_probs = np.array([0.25, 0.25, 0.25, 0.25])

    assert np.isclose(compute_tvd(x_train, samples, exact_probs=exact_probs), 0.5)
    assert np.isclose(compute_tvd(x_train, samples, exact_probs=exact_probs, model_probs=model_probs), 0.5)


def test_visible_wire_marginalization_preserves_bit_order():
    full_probs = np.zeros(8)
    full_probs[0b000] = 0.25
    full_probs[0b101] = 0.75

    visible = marginalize_probs_to_wires(full_probs, n_qubits=3, wires=[0, 2])

    assert np.array_equal(visible, np.array([0.25, 0.0, 0.0, 0.75]))


def test_reorder_probs_to_sample_indexing_bit_reverses_circuit_order():
    circuit_order = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    sample_order = reorder_probs_to_sample_indexing(circuit_order, n_qubits=3)

    assert np.array_equal(sample_order, np.array([0.0, 0.4, 0.2, 0.6, 0.1, 0.5, 0.3, 0.7]))


def test_evaluation_policy_reorders_exact_probs_before_metrics():
    class Circuit:
        n_qubits = 3
        bitflip = False

        def probs(self, params):
            probs = np.zeros(8)
            probs[0b100] = 1.0
            return probs

    policy = EvaluationPolicy(
        x_train=np.array([[1, 0, 0]], dtype=np.int8),
        sigma=1.0,
        shots=2,
        rng_seed=7,
        exact_probs=np.eye(1, 8, 0b001, dtype=np.float64)[0],
        exact_metrics={"enabled": True},
    )

    model_probs = policy.exact_model_probs(Circuit(), np.array([1.0]))
    stats = policy.evaluate_exact_probs(model_probs)

    assert np.array_equal(model_probs, policy.exact_probs)
    assert np.isclose(stats["tvd_exact"], 0.0)


def test_missing_validity_and_coverage_metrics_are_nan():
    x_train = np.array([[0, 0], [1, 1]], dtype=np.int8)
    samples = np.array([[0, 0], [1, 0]], dtype=np.int8)

    stats = evaluate_samples(x_train, samples, sigma=1.0)

    for key in ("validity", "coverage", "precision", "recall", "support_match", "f_score"):
        assert math.isnan(stats[key])


def test_analytical_mmd_returns_nan_for_empty_ensemble():
    class EmptyEnsemble:
        models = []
        weights = []

    x_train = np.array([[0, 0], [1, 1]], dtype=np.int8)

    assert math.isnan(compute_ensemble_training_mmd(EmptyEnsemble(), x_train))


def test_evaluation_policy_sampling_flags_match_config_semantics():
    x_train = np.array([[0, 0]], dtype=np.int8)

    default_policy = EvaluationPolicy(x_train=x_train, sigma=1.0, shots=4, rng_seed=7)
    skipped_policy = EvaluationPolicy(
        x_train=x_train,
        sigma=1.0,
        shots=4,
        rng_seed=7,
        skip_sampling=True,
    )
    final_policy = EvaluationPolicy(
        x_train=x_train,
        sigma=1.0,
        shots=4,
        rng_seed=7,
        skip_sampling=True,
        final_eval_sampling=True,
    )

    assert default_policy.sampling_enabled is True
    assert default_policy.final_sampling_enabled is True
    assert skipped_policy.sampling_enabled is False
    assert skipped_policy.final_sampling_enabled is False
    assert final_policy.sampling_enabled is False
    assert final_policy.final_sampling_enabled is True


def test_sample_and_evaluate_ensemble_uses_deterministic_step_seed():
    class RecordingEnsemble:
        def __init__(self):
            self.samples = None

        def sample(self, n_samples, rng):
            self.samples = rng.integers(0, 2, size=(n_samples, 2), dtype=np.int8)
            return self.samples

    x_train = np.array([[0, 0], [1, 1]], dtype=np.int8)
    policy = EvaluationPolicy(x_train=x_train, sigma=1.0, shots=5, rng_seed=11)
    ensemble = RecordingEnsemble()

    samples, stats = policy.sample_and_evaluate_ensemble(ensemble, step=3)

    expected_rng = np.random.default_rng(11 + 3 * 7919)
    expected = expected_rng.integers(0, 2, size=(5, 2), dtype=np.int8)
    assert np.array_equal(samples, expected)
    assert np.array_equal(ensemble.samples, expected)
    assert "mmd" in stats


def test_evaluation_policy_adds_exact_suffix_metrics_when_enabled():
    class Circuit:
        n_qubits = 2
        bitflip = False

        def probs(self, params):
            return np.array([0.25, 0.25, 0.25, 0.25])

    x_train = np.array([[0, 0], [1, 1]], dtype=np.int8)
    samples = np.array([[0, 0], [0, 0]], dtype=np.int8)
    policy = EvaluationPolicy(
        x_train=x_train,
        sigma=1.0,
        shots=2,
        rng_seed=7,
        exact_probs=np.array([0.5, 0.0, 0.0, 0.5]),
        exact_metrics={"enabled": True},
    )

    model_probs = policy.exact_model_probs(Circuit(), np.array([1.0]))
    stats = policy.evaluate_samples(samples, model_probs=model_probs)

    assert "tvd" in stats
    assert "tvd_exact" in stats
    assert np.isclose(stats["tvd"], 0.5)
    assert np.isclose(stats["tvd_exact"], 0.5)


def test_evaluation_policy_warns_and_skips_exact_metrics_when_unavailable():
    class Circuit:
        n_qubits = 2
        bitflip = True

        def probs(self, params):
            return np.full(4, 0.25)

    policy = EvaluationPolicy(
        x_train=np.array([[0, 0]], dtype=np.int8),
        sigma=1.0,
        shots=2,
        rng_seed=7,
        exact_metrics={"enabled": True},
    )

    with pytest.warns(RuntimeWarning, match="bitflip"):
        assert policy.exact_model_probs(Circuit(), np.array([1.0])) is None
