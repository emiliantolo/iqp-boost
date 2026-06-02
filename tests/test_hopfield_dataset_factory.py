import numpy as np
import pytest

from src.experiment_factory import build_dataset_bundle


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

    assert bundle["dataset_name"] == "Hopfield (6q, 2p)"
    assert bundle["x_train"].shape == (32, 6)
    assert bundle["x_train"].dtype == np.int8
    assert set(np.unique(bundle["x_train"])).issubset({0, 1})
    assert bundle["validity_fn"] is None
    assert bundle["coverage_fn"] is None
    assert bundle["top_k_tvd_fn"] is not None
    assert bundle["custom_viz_fn"] is None
    assert bundle["exact_probs"] is not None
    assert bundle["exact_probs"].shape == (64,)
    assert np.isclose(bundle["exact_probs"].sum(), 1.0)


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

    assert bundle["x_train"].shape == (30, 5)
    assert bundle["x_test"].shape == (10, 5)


@pytest.mark.parametrize("dataset_name", ["bas", "parity", "fashion_mnist", "hamming_ball"])
def test_removed_and_reserved_dataset_keys_raise_clear_error(dataset_name):
    with pytest.raises(ValueError) as exc_info:
        build_dataset_bundle(
            dataset_spec={"name": dataset_name, "params": {}},
            config={"train_samples": 4, "data_seed": 0},
            plot_spec={"kind": "none"},
        )

    message = str(exc_info.value)
    assert f"Unsupported dataset '{dataset_name}'" in message
    assert "Supported datasets: hopfield" in message
    assert "Reserved future datasets: hamming_ball" in message
