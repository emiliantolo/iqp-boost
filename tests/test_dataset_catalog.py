import numpy as np
import pytest

from src import experiment_factory
from src.dataset_catalog import DatasetBundle, SUPPORTED_DATASETS, build_dataset_bundle


def test_supported_datasets_are_catalog_owned():
    assert SUPPORTED_DATASETS == ("hopfield", "hamming_balls")


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
    assert "Supported datasets: hopfield, hamming_balls" in message


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
