import pytest

from src.hpo.trial_config import bodyness_to_sigma, deep_merge, resolve_trial_config, set_nested


class FakeTrial:
    def __init__(self):
        self.user_attrs = {}
        self.calls = []

    def suggest_int(self, name, low, high, step=1):
        self.calls.append(("int", name, low, high, step))
        return low + step

    def suggest_float(self, name, low, high, log=False):
        self.calls.append(("float", name, low, high, log))
        return high if log else low

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, tuple(choices)))
        return choices[-1]

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


def test_set_nested_creates_nested_dicts():
    config = {}
    set_nested(config, "sigma_heuristic.n_sigmas", 4)
    assert config == {"sigma_heuristic": {"n_sigmas": 4}}


def test_deep_merge_preserves_unrelated_defaults():
    merged = deep_merge(
        {"a": {"b": 1, "c": 2}, "unchanged": {"x": 3}},
        {"a": {"b": 9}},
    )
    assert merged == {"a": {"b": 9, "c": 2}, "unchanged": {"x": 3}}


def test_resolve_trial_config_samples_supported_types():
    trial = FakeTrial()
    config = resolve_trial_config(
        {"sigma_heuristic": {"method": "fourier"}},
        {
            "n_models": {"type": "int", "low": 2, "high": 6, "step": 2},
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.05, "log": True},
            "n_ops": {"type": "categorical", "choices": [1000, 2000, 4000]},
        },
        trial,
    )
    assert config["n_models"] == 4
    assert config["learning_rate"] == 0.05
    assert config["n_ops"] == 4000
    assert trial.user_attrs["sampled_config"] == {
        "n_models": 4,
        "learning_rate": 0.05,
        "n_ops": 4000,
    }


def test_resolve_trial_config_samples_continuous_frank_wolfe_schedule():
    trial = FakeTrial()
    config = resolve_trial_config(
        {"lambda_schedule": {"type": "frank_wolfe", "gamma": 0.5, "tau": 1.0}},
        {
            "lambda_schedule.gamma": {"type": "float", "low": 0.01, "high": 1.0, "log": True},
            "lambda_schedule.tau": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        },
        trial,
    )

    assert config["lambda_schedule"] == {
        "type": "frank_wolfe",
        "gamma": 1.0,
        "tau": 10.0,
    }
    assert trial.calls == [
        ("float", "lambda_schedule.gamma", 0.01, 1.0, True),
        ("float", "lambda_schedule.tau", 0.01, 10.0, True),
    ]
    assert trial.user_attrs["sampled_config"] == {
        "lambda_schedule.gamma": 1.0,
        "lambda_schedule.tau": 10.0,
    }


def test_resolve_trial_config_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported search-space type"):
        resolve_trial_config({}, {"x": {"type": "choice"}}, FakeTrial())


def test_resolve_trial_config_samples_constrained_bodyness_sigmas():
    trial = FakeTrial()
    config = resolve_trial_config(
        {"sigma_factor": [0.5], "sigma_heuristic": {"method": "fourier"}},
        {
            "sigma": {
                "type": "bodyness_sigma",
                "n_qubits": 20,
                "n_sigmas_choices": [1, 2, 3],
                "low": 0.5,
                "high": 9.9,
                "min_separation": 1.5,
            }
        },
        trial,
    )

    sampled = trial.user_attrs["sampled_config"]
    targets = sampled["sigma.bodyness_targets"]
    sigmas = sampled["sigma.values"]

    assert sampled["sigma.n_sigmas"] == 3
    assert targets == sorted(targets)
    assert targets[1] - targets[0] >= 1.5
    assert targets[2] - targets[1] >= 1.5
    assert all(value > 0 for value in sigmas)
    assert config["sigma"] == sigmas
    assert "sigma_heuristic" not in config
    assert "sigma_factor" not in config


def test_bodyness_to_sigma_rejects_invalid_targets():
    with pytest.raises(ValueError, match="bodyness target"):
        bodyness_to_sigma(10.0, 20)
