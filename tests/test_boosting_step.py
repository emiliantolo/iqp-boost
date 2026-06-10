import jax
import numpy as np

import src.run.boosting_step as boosting_step
from src.run import BoostingStepContext, run_boosting_step


class FakeEvaluation:
    sampling_enabled = False

    def __init__(self, training_mmd, exact_steps=False):
        self.training_mmd = training_mmd
        self.training_calls = 0
        self.exact_steps = exact_steps

    def evaluate_ensemble_training_mmd(self, ensemble):
        self.training_calls += 1
        return self.training_mmd

    def analytical_mmd_stats(self, value, model_probs=None):
        stats = {"mmd": value}
        if model_probs is not None:
            stats["tvd_exact"] = float(model_probs.sum())
        return stats

    def exact_metrics_enabled(self, phase):
        return self.exact_steps and phase == "steps"

    def exact_ensemble_probs(self, ensemble):
        return np.array([0.5, 0.5])


class FakeEnsemble:
    def __init__(self):
        self.lambda_dual = 1.0
        self.models = [np.array([0.0])]
        self.weights = [1.0]
        self.n_samples = 100
        self.restored = False
        self.snapshots = []

    def snapshot_state(self):
        snapshot = {
            "models": list(self.models),
            "weights": list(self.weights),
            "lambda_dual": self.lambda_dual,
        }
        self.snapshots.append(snapshot)
        return snapshot

    def restore_state(self, snapshot):
        self.restored = True
        self.models = list(snapshot["models"])
        self.weights = list(snapshot["weights"])
        self.lambda_dual = snapshot["lambda_dual"]


def test_accepted_step_appends_history_and_advances_reference(monkeypatch):
    monkeypatch.setattr(boosting_step, "train_candidate_model", _candidate(alpha=0.4))
    ensemble = FakeEnsemble()
    history = {"mmd": [0.8], "step": [0], "alpha": [1.0], "training_loss": [0.8]}

    result = run_boosting_step(_context(ensemble=ensemble, history=history, training_mmd=0.5))

    assert result.accepted is True
    assert result.should_stop is False
    assert result.alpha == 0.4
    assert result.previous_stats == {"mmd": 0.5}
    assert result.previous_training_mmd == 0.5
    assert history == {
        "mmd": [0.8, 0.5],
        "step": [0, 1],
        "alpha": [1.0, 0.4],
        "training_loss": [0.8, 0.5],
        "accepted_by_metric": [True, True],
    }
    assert ensemble.restored is False


def test_rejected_step_restores_snapshot_and_does_not_append_history(monkeypatch):
    monkeypatch.setattr(boosting_step, "train_candidate_model", _candidate(alpha=0.4))
    ensemble = FakeEnsemble()
    history = {"mmd": [0.8], "step": [0], "alpha": [1.0], "training_loss": [0.8]}

    result = run_boosting_step(_context(ensemble=ensemble, history=history, training_mmd=0.9))

    assert result.accepted is False
    assert result.should_stop is False
    assert result.previous_stats == {"mmd": 0.8}
    assert result.previous_training_mmd == 0.8
    assert history == {"mmd": [0.8], "step": [0], "alpha": [1.0], "training_loss": [0.8]}
    assert ensemble.restored is True
    assert len(ensemble.models) == 1


def test_zero_alpha_restores_snapshot_before_evaluation(monkeypatch):
    monkeypatch.setattr(boosting_step, "train_candidate_model", _candidate(alpha=0.0))
    ensemble = FakeEnsemble()
    evaluation = FakeEvaluation(training_mmd=0.5)
    history = {"mmd": [0.8], "step": [0], "alpha": [1.0], "training_loss": [0.8]}

    result = run_boosting_step(
        _context(
            ensemble=ensemble,
            evaluation=evaluation,
            history=history,
            training_mmd=0.5,
            min_alpha_accept=1e-10,
        )
    )

    assert result.accepted is False
    assert result.alpha == 0.0
    assert evaluation.training_calls == 0
    assert history == {"mmd": [0.8], "step": [0], "alpha": [1.0], "training_loss": [0.8]}
    assert ensemble.restored is True


def test_stop_on_reject_sets_should_stop(monkeypatch):
    monkeypatch.setattr(boosting_step, "train_candidate_model", _candidate(alpha=0.4))

    result = run_boosting_step(
        _context(
            training_mmd=0.9,
            config={"stop_on_reject": True},
        )
    )

    assert result.accepted is False
    assert result.should_stop is True


def test_keep_models_for_diagnosis_keeps_worse_step_and_records_flag(monkeypatch):
    monkeypatch.setattr(boosting_step, "train_candidate_model", _candidate(alpha=0.4))
    ensemble = FakeEnsemble()
    history = {"mmd": [0.8], "step": [0], "alpha": [1.0], "training_loss": [0.8]}

    result = run_boosting_step(
        _context(
            ensemble=ensemble,
            history=history,
            training_mmd=0.9,
            config={"keep_models_for_diagnosis": True},
        )
    )

    assert result.accepted is True
    assert result.should_stop is False
    assert result.previous_stats == {"mmd": 0.9}
    assert result.previous_training_mmd == 0.9
    assert history == {
        "mmd": [0.8, 0.9],
        "step": [0, 1],
        "alpha": [1.0, 0.4],
        "training_loss": [0.8, 0.9],
        "accepted_by_metric": [True, False],
    }
    assert ensemble.restored is False
    assert len(ensemble.models) == 2


def test_data_only_step_uses_same_module_without_changing_lambda(monkeypatch):
    monkeypatch.setattr(boosting_step, "train_candidate_model", _candidate(alpha=0.3))
    ensemble = FakeEnsemble()
    ensemble.lambda_dual = 0.0

    result = run_boosting_step(
        _context(
            ensemble=ensemble,
            training_mmd=0.4,
            config={"lambda_dual": 0.0},
            step_prefix="Data-only Step",
            include_sampled_tvd=False,
            apply_lambda_schedule=False,
            compute_snr=False,
        )
    )

    assert result.accepted is True
    assert ensemble.lambda_dual == 0.0


def test_compute_snr_false_skips_gradient_snr_report(monkeypatch):
    monkeypatch.setattr(boosting_step, "train_candidate_model", _candidate(alpha=0.4))

    def fail_report(*args, **kwargs):
        raise AssertionError("SNR diagnostics should be skipped")

    monkeypatch.setattr(boosting_step, "_report_step_snr", fail_report)

    result = run_boosting_step(
        _context(
            training_mmd=0.5,
            config={"compute_snr": False},
            compute_snr=False,
        )
    )

    assert result.accepted is True


def test_cleanup_respects_clear_jax_caches_flag(monkeypatch):
    calls = {"clear": 0, "gc": 0}
    monkeypatch.setattr(boosting_step.jax, "clear_caches", lambda: calls.__setitem__("clear", calls["clear"] + 1))
    monkeypatch.setattr(boosting_step.gc, "collect", lambda: calls.__setitem__("gc", calls["gc"] + 1))

    boosting_step._cleanup_after_step(2, clear_jax_caches=False)

    assert calls == {"clear": 0, "gc": 1}

    boosting_step._cleanup_after_step(2, clear_jax_caches=True)

    assert calls == {"clear": 1, "gc": 2}


def test_step_exact_metrics_are_only_added_when_steps_phase_enabled(monkeypatch):
    monkeypatch.setattr(boosting_step, "train_candidate_model", _candidate(alpha=0.4))
    ensemble = FakeEnsemble()
    history = {"mmd": [0.8], "step": [0], "alpha": [1.0], "training_loss": [0.8]}
    evaluation = FakeEvaluation(training_mmd=0.5, exact_steps=True)

    def analytical_with_exact(value, model_probs=None):
        stats = {"mmd": value}
        if model_probs is not None:
            stats["tvd_exact"] = float(model_probs.sum())
        return stats

    evaluation.analytical_mmd_stats = analytical_with_exact
    evaluation.exact_ensemble_probs = lambda ensemble: np.array([0.5, 0.5])

    run_boosting_step(_context(ensemble=ensemble, evaluation=evaluation, history=history, training_mmd=0.5))

    assert np.isnan(history["tvd_exact"][0])
    assert history["tvd_exact"][1] == 1.0


def _candidate(alpha):
    def fake_candidate(context):
        context.ensemble.models.append(np.array([context.step], dtype=float))
        context.ensemble.weights.append(alpha)
        return context.key, alpha

    return fake_candidate


def _context(
    *,
    ensemble=None,
    evaluation=None,
    history=None,
    training_mmd=0.5,
    config=None,
    min_alpha_accept=1e-10,
    step_prefix="Step",
    include_sampled_tvd=True,
    compute_snr=False,
    apply_lambda_schedule=True,
):
    config = {
        "n_models": 3,
        "lambda_dual": 1.0,
        "weight_strategy": "frank_wolfe",
        **(config or {}),
    }
    return BoostingStepContext(
        config=config,
        ensemble=ensemble or FakeEnsemble(),
        x_train=np.array([[0, 0], [1, 1]], dtype=np.int8),
        key=jax.random.PRNGKey(0),
        evaluation=evaluation or FakeEvaluation(training_mmd),
        metrics_history=history or {"mmd": [0.8], "step": [0], "alpha": [1.0], "training_loss": [0.8]},
        step=1,
        monitor_interval=None,
        turbo_opt=None,
        acceptance_metric="training_mmd",
        min_alpha_accept=min_alpha_accept,
        previous_stats={"mmd": 0.8},
        previous_training_mmd=0.8,
        step_prefix=step_prefix,
        include_sampled_tvd=include_sampled_tvd,
        apply_lambda_schedule=apply_lambda_schedule,
        compute_snr=compute_snr,
        run_cleanup=False,
    )
