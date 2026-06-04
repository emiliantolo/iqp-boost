import numpy as np
import pytest

from src.datasets.base import BinaryDataset


class CountingDataset(BinaryDataset):
    def _generate_samples(self, n_samples: int, seed: int = 0) -> np.ndarray:
        data = np.arange(n_samples * 2, dtype=np.int64).reshape(n_samples, 2)
        return (data % 2).astype(np.int8)

    def visualize(self, sample: np.ndarray, ax=None):
        return ax


def test_base_dataset_generate_without_split():
    dataset = CountingDataset()

    data = dataset.generate(n_samples=5, seed=3)

    assert data.shape == (5, 2)
    assert dataset.data is data
    assert dataset.active_split == "train"


def test_base_dataset_split_returns_stable_train_test_and_all_slices():
    dataset = CountingDataset(train_split_ratio=0.6)

    x_train = dataset.generate(n_samples=10, seed=3, split="train")
    x_test = dataset.generate(split="test")
    x_all = dataset.generate(split="all")

    assert x_train.shape == (6, 2)
    assert x_test.shape == (4, 2)
    assert x_all.shape == (10, 2)
    assert np.array_equal(x_all[:6], x_train)
    assert np.array_equal(x_all[6:], x_test)


def test_base_dataset_rejects_invalid_split_name():
    dataset = CountingDataset()

    with pytest.raises(ValueError, match="split must be"):
        dataset.generate(n_samples=5, split="validation")


def test_base_dataset_rejects_invalid_split_ratio():
    with pytest.raises(ValueError, match="train_split_ratio must be"):
        CountingDataset(train_split_ratio=1.0)


def test_base_dataset_rejects_empty_split_outcomes():
    dataset = CountingDataset(train_split_ratio=0.9)

    with pytest.raises(ValueError, match="empty train or test split"):
        dataset.generate(n_samples=1)
