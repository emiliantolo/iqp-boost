import numpy as np
import pytest

import src.experiment_factory as experiment_factory
from src.datasets.barabasi_albert_graph import BarabasiAlbertGraphDataset
from src.datasets.bas import VariableLengthBarsAndStripesDataset
from src.datasets.fashion_mnist import FashionMNISTDownscaledDataset
from src.datasets.k_body_parity import KBodyParityDataset


def test_barabasi_albert_reproducibility_shape_and_graph_properties():
    first = BarabasiAlbertGraphDataset(
        nodes=6,
        m=2,
        n_graphs=12,
        train_split_ratio=0.75,
        seed=7,
        store_graphs=True,
    )
    second = BarabasiAlbertGraphDataset(
        nodes=6,
        m=2,
        n_graphs=12,
        train_split_ratio=0.75,
        seed=7,
        store_graphs=True,
    )

    assert first.all_data.shape == (12, 15)
    assert first.train_data.shape == (9, 15)
    assert first.test_data.shape == (3, 15)
    assert first.data.dtype == np.int8
    np.testing.assert_array_equal(first.all_data, second.all_data)
    assert len(first.graphs) == 12
    assert all(graph.number_of_nodes() == 6 for graph in first.graphs)
    assert all(graph.number_of_edges() == 8 for graph in first.graphs)


def test_barabasi_albert_generate_and_evaluate():
    dataset = BarabasiAlbertGraphDataset(nodes=6, m=2, n_graphs=12, train_split_ratio=0.75, seed=7)

    sampled_first = dataset.generate(n_samples=5, seed=123, split="test")
    sampled_second = dataset.generate(n_samples=5, seed=123, split="test")
    np.testing.assert_array_equal(sampled_first, sampled_second)

    metrics = dataset.evaluate_generation(dataset.train_data[:3])
    assert metrics["valid_graph_rate"] == 1.0
    assert metrics["memorization_rate"] == 1.0
    assert "ba_degree_histogram_distance" in metrics


def test_barabasi_albert_invalid_params():
    with pytest.raises(ValueError, match="m must be positive"):
        BarabasiAlbertGraphDataset(nodes=6, m=0)
    with pytest.raises(ValueError, match="m must be smaller"):
        BarabasiAlbertGraphDataset(nodes=6, m=6)


def test_k_body_parity_generation_validity_and_explicit_indices():
    dataset = KBodyParityDataset(
        n_qubits=6,
        k=3,
        parity=1,
        hidden_indices=[0, 2, 4],
        max_exact_states=128,
    )

    samples = dataset.generate(n_samples=64, seed=5)
    assert samples.shape == (64, 6)
    assert samples.dtype == np.int8
    assert dataset.validity_rate(samples) == 1.0
    assert dataset.hidden_indices == (0, 2, 4)
    assert dataset.probs is not None
    assert np.isclose(dataset.probs.sum(), 1.0)


def test_k_body_parity_reproducibility_and_validation_errors():
    first = KBodyParityDataset(n_qubits=8, k=4, parity=0, subset_seed=3)
    second = KBodyParityDataset(n_qubits=8, k=4, parity=0, subset_seed=3)
    np.testing.assert_array_equal(first.generate(20, seed=9), second.generate(20, seed=9))

    with pytest.raises(ValueError, match="k must be less"):
        KBodyParityDataset(n_qubits=4, k=5)
    with pytest.raises(ValueError, match="hidden_indices length"):
        KBodyParityDataset(n_qubits=4, k=2, hidden_indices=[0])


def test_variable_bas_support_generation_and_validity():
    dataset = VariableLengthBarsAndStripesDataset(
        height=4,
        width=4,
        min_length=1,
        max_length=2,
        max_segments=2,
    )
    samples = dataset.generate(n_samples=40, seed=12)

    assert samples.shape == (40, 16)
    assert samples.dtype == np.int8
    assert dataset.validity_rate(samples) == 1.0

    row_and_column = np.zeros((4, 4), dtype=np.int8)
    row_and_column[1, :] = 1
    row_and_column[:, 2] = 1
    assert tuple(row_and_column.reshape(-1).tolist()) in dataset._valid_patterns


def test_variable_bas_reproducibility_and_validation_errors():
    first = VariableLengthBarsAndStripesDataset(height=3, width=5, min_length=2, max_length=3, max_segments=2)
    second = VariableLengthBarsAndStripesDataset(height=3, width=5, min_length=2, max_length=3, max_segments=2)
    np.testing.assert_array_equal(first.generate(25, seed=4), second.generate(25, seed=4))

    with pytest.raises(ValueError, match="max_length"):
        VariableLengthBarsAndStripesDataset(height=3, width=3, min_length=3, max_length=2)
    with pytest.raises(ValueError, match="no valid"):
        VariableLengthBarsAndStripesDataset(height=2, width=2, min_length=3, max_length=4)


def test_fashion_mnist_downscaled_uses_mocked_loader(monkeypatch):
    reduced = (np.arange(10 * 16).reshape(10, 16) % 2).astype(np.int8)

    def fake_load(self):
        return reduced

    monkeypatch.setattr(FashionMNISTDownscaledDataset, "_load_reduced", fake_load)
    dataset = FashionMNISTDownscaledDataset(rows=4, cols=4, threshold=0.4)

    first = dataset.generate(n_samples=6, seed=14)
    second = dataset.generate(n_samples=6, seed=14)
    assert first.shape == (6, 16)
    assert first.dtype == np.int8
    np.testing.assert_array_equal(first, second)


def test_new_datasets_are_config_manageable(monkeypatch):
    reduced = (np.arange(20 * 9).reshape(20, 9) % 2).astype(np.int8)
    monkeypatch.setattr(FashionMNISTDownscaledDataset, "_load_reduced", lambda self: reduced)

    specs = [
        {"name": "barabasi_albert_graph", "params": {"nodes": 6, "m": 2, "n_graphs": 12, "train_split_ratio": 0.75, "seed": 7}},
        {"name": "k_body_parity", "params": {"n_qubits": 6, "k": 3, "parity": 1, "hidden_indices": [0, 2, 4]}},
        {"name": "variable_bas", "params": {"dims": [4, 4], "min_length": 1, "max_length": 2, "max_segments": 2}},
        {"name": "fashion_mnist", "params": {"dims": [3, 3], "threshold": 0.5}},
    ]

    for spec in specs:
        bundle = experiment_factory.build_dataset_bundle(
            dataset_spec=spec,
            config={"train_samples": 8, "data_seed": 11, "dims": [4, 4]},
            plot_spec={"kind": "none"},
        )
        assert bundle["x_train"].shape[0] > 0
        assert bundle["x_train"].dtype == np.int8
        assert bundle["dataset_name"]
