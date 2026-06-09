import json
from pathlib import Path
import numpy as np
import pytest
import inspect
import torch

from src.experiments import factory as experiment_factory
from src.datasets import DatasetBundle, SUPPORTED_DATASETS, build_dataset_bundle
from src.run import run_boosting_experiment


def test_supported_datasets_are_catalog_owned():
    assert SUPPORTED_DATASETS == ("hopfield", "hamming_balls", "mnist")


def test_hopfield_bundle_builds_binary_training_data_and_exact_probs():
    bundle = build_dataset_bundle(
        dataset_spec={
            "name": "hopfield",
            "params": {
                "n_qubits": 6,
                "n_patterns": 2,
                "beta": 1.5,
                "pattern_seed": 3,
                "max_exact_states": 128,
            },
        },
        config={"train_samples": 32, "data_seed": 11},
        plot_spec={"kind": "none"},
    )

    assert isinstance(bundle, DatasetBundle)
    assert bundle.dataset_name == "Hopfield (6q, 2p)"
    assert bundle.n_qubits == 6
    assert bundle.x_train.shape == (32, 6)
    assert bundle.x_train.dtype == np.int8
    assert set(np.unique(bundle.x_train)).issubset({0, 1})
    assert bundle.validity_fn is None
    assert bundle.coverage_fn is None
    assert bundle.top_k_tvd_fn is not None
    assert bundle.custom_viz_fn is None
    assert bundle.exact_probs is not None
    assert bundle.exact_probs.shape == (64,)
    assert np.isclose(bundle.exact_probs.sum(), 1.0)


def test_bundle_builds_evaluation_policy_with_dataset_capabilities():
    exact_probs = np.full(4, 0.25)

    def validity_fn(samples):
        return 1.0

    def coverage_fn(ground_truth, samples):
        return 0.5

    def generation_eval_fn(samples):
        return {"custom_metric": 0.25}

    bundle = DatasetBundle(
        dataset_name="Synthetic",
        x_train=np.array([[0, 0], [1, 1]], dtype=np.int8),
        validity_fn=validity_fn,
        coverage_fn=coverage_fn,
        exact_probs=exact_probs,
        generation_eval_fn=generation_eval_fn,
    )

    policy = bundle.build_evaluation_policy(
        sigma=1.0,
        shots=16,
        rng_seed=3,
        skip_sampling=True,
        final_eval_sampling=True,
    )

    assert policy.x_train is bundle.x_train
    assert policy.validity_fn is validity_fn
    assert policy.coverage_fn is coverage_fn
    assert policy.exact_probs is exact_probs
    assert policy.generation_eval_fn is generation_eval_fn
    assert policy.sampling_enabled is False
    assert policy.final_sampling_enabled is True


def test_runner_interface_accepts_dataset_bundle_not_scattered_dataset_kwargs():
    params = inspect.signature(run_boosting_experiment).parameters

    assert "dataset" in params
    assert params["dataset"].annotation is DatasetBundle
    assert "dataset_name" not in params
    assert "x_train" not in params
    assert "validity_fn" not in params
    assert "coverage_fn" not in params
    assert "exact_probs" not in params
    assert "custom_viz_fn" not in params


def test_hopfield_bundle_supports_train_test_split():
    bundle = build_dataset_bundle(
        dataset_spec={
            "name": "hopfield",
            "params": {
                "n_qubits": 5,
                "n_patterns": 2,
                "test_samples": 10,
                "train_split_ratio": 0.75,
                "max_exact_states": 64,
            },
        },
        config={"train_samples": 30, "data_seed": 7},
        plot_spec={"kind": "none"},
    )

    assert bundle.x_train.shape == (30, 5)
    assert bundle.x_test is not None
    assert bundle.x_test.shape == (10, 5)


def test_hamming_balls_bundle_supports_train_test_split():
    bundle = build_dataset_bundle(
        dataset_spec={
            "name": "hamming_balls",
            "params": {
                "n_qubits": 5,
                "K": 3,
                "test_samples": 10,
                "max_exact_states": 64,
            },
        },
        config={"train_samples": 30, "data_seed": 7},
        plot_spec={"kind": "none"},
    )

    assert bundle.x_train.shape == (30, 5)
    assert bundle.x_test is not None
    assert bundle.x_test.shape == (10, 5)


def test_hamming_balls_bundle_builds_binary_training_data_and_exact_probs():
    bundle = build_dataset_bundle(
        dataset_spec={
            "name": "hamming_balls",
            "params": {
                "n_qubits": 6,
                "K": 3,
                "p": 0.1,
                "pattern_seed": 5,
                "max_exact_states": 128,
            },
        },
        config={"train_samples": 32, "data_seed": 13},
        plot_spec={"kind": "none"},
    )

    assert bundle.dataset_name == "Hamming Balls (n=6, K=3, p=0.1)"
    assert bundle.x_train.shape == (32, 6)
    assert bundle.x_train.dtype == np.int8
    assert set(np.unique(bundle.x_train)).issubset({0, 1})
    assert bundle.validity_fn is None
    assert bundle.coverage_fn is None
    assert bundle.custom_viz_fn is None
    assert bundle.exact_probs is not None
    assert bundle.exact_probs.shape == (64,)
    assert np.isclose(bundle.exact_probs.sum(), 1.0)
    assert bundle.generation_eval_fn is None
    assert bundle.x_test is None


def test_hamming_balls_default_plot_kind_is_supported_noop():
    bundle = build_dataset_bundle(
        dataset_spec={
            "name": "hamming_balls",
            "params": {"n_qubits": 5, "K": 3, "p": 0.1, "max_exact_states": 64},
        },
        config={"train_samples": 8, "data_seed": 13},
    )

    assert bundle.custom_viz_fn is None


def _patch_fake_mnist(monkeypatch):
    import torchvision.datasets

    class FakeMNIST:
        def __init__(self, root, train=True, download=True):
            n_samples = 30 if train else 18
            base = torch.linspace(0, 255, steps=n_samples * 28 * 28, dtype=torch.float32)
            self.data = base.reshape(n_samples, 28, 28).to(torch.uint8)
            self.targets = torch.tensor([label % 10 for label in range(n_samples)])

    monkeypatch.setattr(torchvision.datasets, "MNIST", FakeMNIST)


def test_mnist_bundle_builds_10x10_binary_training_data(monkeypatch):
    _patch_fake_mnist(monkeypatch)

    bundle = build_dataset_bundle(
        dataset_spec={
            "name": "mnist",
            "params": {
                "rows": 10,
                "cols": 10,
                "threshold": 0.4,
                "classes": [0, 1, 2, 3],
                "data_dir": "./data",
            },
        },
        config={"train_samples": 12, "data_seed": 5},
        plot_spec={"kind": "none"},
    )

    assert bundle.dataset_name == "MNIST (10x10, 4 classes, threshold=0.4)"
    assert bundle.n_qubits == 100
    assert bundle.x_train.shape == (12, 100)
    assert bundle.x_train.dtype == np.int8
    assert set(np.unique(bundle.x_train)).issubset({0, 1})
    assert bundle.exact_probs is None
    assert bundle.custom_viz_fn is None
    assert bundle.dataset_obj.threshold == 0.4
    assert bundle.dataset_obj.classes == [0, 1, 2, 3]


def test_mnist_bundle_uses_test_split_when_requested(monkeypatch):
    _patch_fake_mnist(monkeypatch)

    bundle = build_dataset_bundle(
        dataset_spec={
            "name": "mnist",
            "params": {
                "rows": 10,
                "cols": 10,
                "threshold": 0.4,
                "classes": None,
                "test_samples": 9,
            },
        },
        config={"train_samples": 11, "data_seed": 7},
        plot_spec={"kind": "none"},
    )

    assert bundle.x_train.shape == (11, 100)
    assert bundle.x_test is not None
    assert bundle.x_test.shape == (9, 100)
    assert bundle.x_test.dtype == np.int8
    assert set(np.unique(bundle.x_test)).issubset({0, 1})


def test_mnist_hpo_configs_optimize_test_mmd_and_request_test_samples():
    config_paths = sorted(Path("configs/hpo/mnist").glob("mnist_100q_*class.json"))
    assert {path.stem for path in config_paths} == {
        "mnist_100q_1class",
        "mnist_100q_2class",
        "mnist_100q_4class",
        "mnist_100q_6class",
        "mnist_100q_10class",
    }

    for path in config_paths:
        spec = json.loads(path.read_text())
        params = spec["dataset"]["params"]
        assert spec["objective_metric"] == "test_mmd"
        assert spec["dataset"]["name"] == "mnist"
        assert params["rows"] == 10
        assert params["cols"] == 10
        assert params["threshold"] == 0.4
        assert params["test_samples"] > 0
        assert spec["fixed_config"]["sigma"] == [5.0, 2.5, 1.5, 1.0]
        assert spec["fixed_config"]["n_models"] == 10
        assert spec["fixed_config"]["n_samples"] == 2048
        assert "n_ops" not in spec["fixed_config"]
        assert "sigma_heuristic" not in spec["fixed_config"]
        assert "n_models" not in spec["search_space"]
        assert spec["search_space"]["learning_rate"] == {
            "type": "float",
            "low": 0.001,
            "high": 0.1,
            "log": True,
        }
        assert spec["search_space"]["dynamic_is"] == {
            "type": "categorical",
            "choices": [False, True],
        }
        assert spec["search_space"]["dynamic_is_beta"] == {
            "type": "float",
            "low": 0.01,
            "high": 0.2,
            "log": True,
        }
        assert spec["search_space"]["lambda_schedule.gamma"] == {
            "type": "categorical",
            "choices": [0.25, 0.5, 0.75, 1.0],
        }
        assert spec["search_space"]["n_ops"] == {
            "type": "categorical",
            "choices": [2048, 4096, 8192],
        }
        assert spec["best_retrains"] == {
            "n_seeds": 5,
            "seed_start": 0,
            "baseline": "standalone",
            "report_fcfw": True,
            "skip_sampling": True,
            "final_eval_sampling": False,
        }


@pytest.mark.parametrize("dataset_name", ["bas", "parity", "fashion_mnist"])
def test_removed_and_reserved_dataset_keys_raise_clear_error(dataset_name):
    with pytest.raises(ValueError) as exc_info:
        build_dataset_bundle(
            dataset_spec={"name": dataset_name, "params": {}},
            config={"train_samples": 4, "data_seed": 0},
            plot_spec={"kind": "none"},
        )

    message = str(exc_info.value)
    assert f"Unsupported dataset '{dataset_name}'" in message
    assert "Supported datasets: hopfield, hamming_balls, mnist" in message


def test_singular_hamming_ball_key_points_to_supported_name():
    with pytest.raises(ValueError, match="Use 'hamming_balls'"):
        build_dataset_bundle(
            dataset_spec={"name": "hamming_ball", "params": {}},
            config={"train_samples": 4, "data_seed": 0},
            plot_spec={"kind": "none"},
        )


def test_experiment_factory_delegates_to_dataset_catalog():
    bundle = experiment_factory.build_dataset_bundle(
        dataset_spec={"name": "hopfield", "params": {"n_qubits": 4, "max_exact_states": 32}},
        config={"train_samples": 4, "data_seed": 0},
        plot_spec={"kind": "none"},
    )

    assert isinstance(bundle, DatasetBundle)
    assert experiment_factory.SUPPORTED_DATASETS == SUPPORTED_DATASETS
