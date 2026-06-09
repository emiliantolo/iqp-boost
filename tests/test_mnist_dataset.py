import numpy as np
import pytest
import torch

from src.datasets.mnist import MNISTDataset


def _image_for_label(label: int) -> torch.Tensor:
    bits = torch.tensor([(label >> bit) & 1 for bit in range(10)], dtype=torch.uint8)
    columns = bits.repeat_interleave(3)[:28]
    return (columns.unsqueeze(0).repeat(28, 1) * 255).to(torch.uint8)


def _patch_labeled_mnist(monkeypatch, train_labels, test_labels=None):
    import torchvision.datasets

    test_labels = train_labels if test_labels is None else test_labels

    class FakeMNIST:
        def __init__(self, root, train=True, download=True):
            labels = train_labels if train else test_labels
            self.targets = torch.tensor(labels, dtype=torch.int64)
            self.data = torch.stack([_image_for_label(int(label)) for label in labels])

    monkeypatch.setattr(torchvision.datasets, "MNIST", FakeMNIST)


def _decoded_counts(samples: np.ndarray) -> dict[int, int]:
    powers = 2 ** np.arange(10)
    labels = samples.astype(int) @ powers
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def test_generate_balanced_returns_equal_count_per_requested_class(monkeypatch):
    _patch_labeled_mnist(monkeypatch, train_labels=[0, 1, 2] * 4)

    dataset = MNISTDataset(rows=1, cols=10, threshold=0.5, classes=[0, 1, 2])
    samples = dataset.generate_balanced(n_per_class=3, seed=5, split="train")

    assert samples.shape == (9, 10)
    assert _decoded_counts(samples) == {0: 3, 1: 3, 2: 3}
    assert dataset.active_split == "train"


def test_generate_balanced_is_deterministically_shuffled(monkeypatch):
    _patch_labeled_mnist(monkeypatch, train_labels=[0, 1, 2] * 4)

    dataset = MNISTDataset(rows=1, cols=10, threshold=0.5, classes=[0, 1, 2])
    first = dataset.generate_balanced(n_per_class=2, seed=7, split="train")
    second = dataset.generate_balanced(n_per_class=2, seed=7, split="train")
    third = dataset.generate_balanced(n_per_class=2, seed=8, split="train")

    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)
    assert _decoded_counts(first) == {0: 2, 1: 2, 2: 2}


def test_generate_balanced_allows_replacement_when_class_request_exceeds_available(monkeypatch):
    _patch_labeled_mnist(monkeypatch, train_labels=[3, 3])

    dataset = MNISTDataset(rows=1, cols=10, threshold=0.5, classes=[3])
    samples = dataset.generate_balanced(n_per_class=5, seed=11, split="train")

    assert samples.shape == (5, 10)
    assert _decoded_counts(samples) == {3: 5}


def test_generate_balanced_rejects_all_split(monkeypatch):
    _patch_labeled_mnist(monkeypatch, train_labels=[0, 1])

    dataset = MNISTDataset(rows=1, cols=10, threshold=0.5, classes=[0, 1])
    with pytest.raises(ValueError, match="split='train' or split='test'"):
        dataset.generate_balanced(n_per_class=1, split="all")
