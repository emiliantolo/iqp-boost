import numpy as np
import pytest

import src.core.weight_strategy as weight_strategy
from src.core import WeightStrategyContext, apply_weight_strategy


class FakeTerms:
    def __init__(self):
        self.trs = [["m0"], ["m1"]]
        self.corrs = [["c0"], ["c1"]]


class FakeEnsemble:
    def __init__(self, weights=None):
        self.weights = list(weights or [0.6, 0.4])
        self.sigma = 1.5
        self.n_samples = 11
        self.terms = FakeTerms()

    def normalize_weights(self):
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]


def test_greedy_and_frank_wolfe_return_newest_weight_without_mutation():
    ensemble = FakeEnsemble([0.7, 0.3])

    greedy = apply_weight_strategy(WeightStrategyContext(ensemble=ensemble, strategy="greedy"))
    frank_wolfe = apply_weight_strategy(WeightStrategyContext(ensemble=ensemble, strategy="frank_wolfe"))

    assert greedy.alpha == 0.3
    assert frank_wolfe.alpha == 0.3
    assert ensemble.weights == [0.7, 0.3]
    assert np.array_equal(greedy.weights, np.array([0.7, 0.3]))


def test_alpha_strategy_rescales_previous_weights_once(monkeypatch):
    monkeypatch.setattr(weight_strategy, "compute_optimal_alpha_tvd_samples", lambda **kwargs: 0.25)
    ensemble = FakeEnsemble([0.2, 0.3, 0.5])

    result = apply_weight_strategy(WeightStrategyContext(
        ensemble=ensemble,
        strategy="tvd_line_search",
        ground_truth=np.array([[0, 0]], dtype=np.int8),
        samples_old=np.array([[0, 0]], dtype=np.int8),
        samples_new=np.array([[1, 1]], dtype=np.int8),
    ))

    assert result.alpha == 0.25
    assert np.allclose(ensemble.weights, [0.3, 0.45, 0.25])
    assert np.allclose(result.weights, [0.3, 0.45, 0.25])
    assert result.fallback_to_greedy is False


def test_missing_required_inputs_fall_back_to_greedy_and_keep_weights():
    ensemble = FakeEnsemble([0.7, 0.3])

    result = apply_weight_strategy(WeightStrategyContext(ensemble=ensemble, strategy="validity_line_search"))

    assert result.alpha == 0.3
    assert result.fallback_to_greedy is True
    assert "Missing samples/validity_fn" in result.warning
    assert ensemble.weights == [0.7, 0.3]


def test_fully_corrective_replaces_weights_with_qp_result(monkeypatch):
    calls = []

    def fake_qp(**kwargs):
        calls.append(kwargs)
        return np.array([0.1, 0.9])

    monkeypatch.setattr(weight_strategy, "compute_optimal_weights_qp", fake_qp)
    ensemble = FakeEnsemble([0.5, 0.5])

    result = apply_weight_strategy(WeightStrategyContext(
        ensemble=ensemble,
        strategy="fully_corrective",
        trs_data=["data"],
    ))

    assert result.alpha == 0.9
    assert np.allclose(ensemble.weights, [0.1, 0.9])
    assert np.allclose(result.weights, [0.1, 0.9])
    assert calls == [{
        "all_trs": ensemble.terms.trs,
        "trs_data": ["data"],
        "all_corrs": ensemble.terms.corrs,
        "n_samples": ensemble.n_samples,
    }]


def test_unknown_strategy_raises_value_error():
    with pytest.raises(ValueError, match="Unknown weight_strategy"):
        apply_weight_strategy(WeightStrategyContext(ensemble=FakeEnsemble(), strategy="mystery"))
