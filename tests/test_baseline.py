import jax
import numpy as np

import src.run.baseline as baseline
import src.run.runner as runner
from src.run import BaselineContext, BaselineResult, resolve_baselines_to_run, run_baselines
from src.datasets import DatasetBundle


class FakeEvaluation:
    def __init__(self, final_sampling_enabled=False):
        self.final_sampling_enabled = final_sampling_enabled
        self.sampled_circuits = []
        self.sampled_ensembles = []

    def sample_and_evaluate_circuit(self, circuit, params, wires=None):
        self.sampled_circuits.append((circuit, params, wires))
        return np.ones((2, 2), dtype=np.int8), {"mmd": 0.25}

    def sample_and_evaluate_ensemble(self, ensemble, step):
        self.sampled_ensembles.append((ensemble, step))
        return np.ones((2, 2), dtype=np.int8), {"mmd": 0.33}

    def analytical_mmd_stats(self, value, model_probs=None):
        stats = {"mmd": value}
        if model_probs is not None:
            stats.update(self.evaluate_exact_probs(model_probs))
        return stats

    def exact_metrics_enabled(self, phase):
        return phase == "baseline"

    def exact_model_probs(self, circuit, params, wires=None):
        return np.array([0.5, 0.5])

    def evaluate_exact_probs(self, model_probs):
        return {"tvd_exact": float(np.asarray(model_probs).sum())}


class FakeTrainer:
    losses = [0.9, 0.7]
    final_params = np.array([1.0])


class FakeEnsemble:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.lambda_dual = kwargs["lambda_dual"]
        self.models = []
        self.weights = []


def test_baseline_none_values_disable_baselines():
    assert resolve_baselines_to_run({"baseline": "none"}) == []
    assert resolve_baselines_to_run({"baseline": None}) == []


def test_legacy_baseline_config_is_preserved_when_baseline_key_is_absent():
    assert resolve_baselines_to_run({"baselines_to_run": ["data_only"]}) == ["data_only"]
    assert resolve_baselines_to_run({"run_data_only_baseline": True}) == ["standalone", "data_only"]
    assert resolve_baselines_to_run({}) == ["standalone"]
    assert resolve_baselines_to_run({"baseline": []}) == ["standalone"]


def test_standalone_analytical_path_returns_stats_losses_params_and_no_samples(monkeypatch):
    monkeypatch.setattr(baseline, "train_standalone_model", lambda context, key, epochs: (key, FakeTrainer(), "params"))
    monkeypatch.setattr(baseline, "report_baseline", lambda *args, **kwargs: None)

    result = run_baselines(_context(config={"baseline": "standalone"}, final_sampling_enabled=False))

    assert result.selected_baselines == ["standalone"]
    assert result.standalone_stats == {"mmd": 0.7, "tvd_exact": 1.0, "training_loss": 0.7}
    assert result.standalone_samples is None
    assert result.standalone_params == "params"
    assert result.standalone_train_losses == [0.9, 0.7]
    assert result.reference_label == "Standalone"
    assert result.reference_stats == result.standalone_stats


def test_standalone_sampled_path_returns_samples_and_preserves_training_loss(monkeypatch):
    monkeypatch.setattr(baseline, "train_standalone_model", lambda context, key, epochs: (key, FakeTrainer(), "params"))
    monkeypatch.setattr(baseline, "report_baseline", lambda *args, **kwargs: None)
    evaluation = FakeEvaluation(final_sampling_enabled=True)

    result = run_baselines(_context(config={"baseline": "standalone"}, evaluation=evaluation))

    assert np.array_equal(result.standalone_samples, np.ones((2, 2), dtype=np.int8))
    assert result.standalone_stats == {"mmd": 0.25, "training_loss": 0.7}
    assert evaluation.sampled_circuits == [("circuit", "params", None)]


def test_data_only_path_delegates_to_boosting_step_with_zero_lambda(monkeypatch):
    calls = []

    def fake_initialize(**kwargs):
        calls.append(("initialize", kwargs["config"]["lambda_dual"], kwargs["include_sampled_tvd"]))
        return _initial_result()

    def fake_step(context):
        calls.append((
            "step",
            context.config["lambda_dual"],
            context.apply_lambda_schedule,
            context.compute_snr,
            context.run_cleanup,
            context.step_prefix,
        ))
        return _step_result(context)

    monkeypatch.setattr(baseline, "BoostedEnsemble", FakeEnsemble)
    monkeypatch.setattr(baseline, "initialize_boosting_ensemble", fake_initialize)
    monkeypatch.setattr(baseline, "run_boosting_step", fake_step)
    monkeypatch.setattr(baseline, "report_baseline", lambda *args, **kwargs: None)

    result = run_baselines(_context(config={"baseline": "data_only", "n_models": 2}))

    assert result.selected_baselines == ["data_only"]
    assert result.data_only_stats == {"mmd": 0.4}
    assert result.data_only_history["training_loss"] == [0.8, 0.4]
    assert result.data_only_ensemble.lambda_dual == 0.0
    assert calls == [
        ("initialize", 0.0, False),
        ("step", 0.0, False, False, False, "Data-only Step"),
    ]


def test_reference_selection_prefers_standalone_then_data_only_and_falls_back_to_synthetic():
    both = BaselineResult(
        key=jax.random.PRNGKey(0),
        selected_baselines=["standalone", "data_only"],
        standalone_stats={"mmd": 0.1},
        data_only_stats={"mmd": 0.2},
        reference_label="Standalone",
        reference_stats={"mmd": 0.1},
    )
    assert both.reference_or_synthetic(0.9) == ("Standalone", {"mmd": 0.1})

    data_only = BaselineResult(
        key=jax.random.PRNGKey(0),
        selected_baselines=["data_only"],
        data_only_stats={"mmd": 0.2},
        reference_label="Data-only",
        reference_stats={"mmd": 0.2},
    )
    assert data_only.reference_or_synthetic(0.9) == ("Data-only", {"mmd": 0.2})

    none = BaselineResult(key=jax.random.PRNGKey(0), selected_baselines=[])
    assert none.reference_or_synthetic(0.9) == ("Synthetic", {"mmd": 0.9})


def test_runner_no_longer_exports_baseline_helpers():
    assert not hasattr(runner, "train_standalone_model")
    assert not hasattr(runner, "run_data_only_ensemble_baseline")
    assert not hasattr(runner, "_resolve_baselines_to_run")


def _context(*, config=None, evaluation=None, final_sampling_enabled=False):
    cfg = {
        "baseline": "none",
        "n_models": 1,
        "epochs_per_step": 2,
        "learning_rate": 0.01,
        **(config or {}),
    }
    return BaselineContext(
        config=cfg,
        circuit="circuit",
        dataset=DatasetBundle("Synthetic", np.array([[0, 0]], dtype=np.int8)),
        x_train=np.array([[0, 0]], dtype=np.int8),
        key=jax.random.PRNGKey(0),
        sigma=1.0,
        n_ops=4,
        n_samples=8,
        wires=None,
        monitor_interval=None,
        turbo_opt=None,
        evaluation=evaluation or FakeEvaluation(final_sampling_enabled=final_sampling_enabled),
        acceptance_metric="training_mmd",
        min_alpha_accept=1e-10,
    )


def _initial_result():
    return type(
        "Initial",
        (),
        {
            "key": jax.random.PRNGKey(1),
            "metrics_history": {"mmd": [0.8], "step": [0], "alpha": [1.0], "training_loss": [0.8]},
            "stats": {"mmd": 0.8},
            "training_mmd": 0.8,
        },
    )()


def _step_result(context):
    context.metrics_history["mmd"].append(0.4)
    context.metrics_history["step"].append(1)
    context.metrics_history["alpha"].append(0.5)
    context.metrics_history["training_loss"].append(0.4)
    return type(
        "Step",
        (),
        {
            "key": jax.random.PRNGKey(2),
            "should_stop": False,
            "previous_stats": {"mmd": 0.4},
            "previous_training_mmd": 0.4,
        },
    )()
