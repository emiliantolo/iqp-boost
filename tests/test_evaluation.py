import math

import numpy as np

from src.evaluation import EvaluationPolicy, compute_ensemble_training_mmd, evaluate_samples


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
