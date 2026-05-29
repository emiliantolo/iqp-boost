from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from src.datasets._registry import Dataset, DatasetBundle, EvaluationCapabilities, register_builder


def bars_and_stripes(
    height: int = 4,
    width: int = 4,
    flatten: bool = True,
) -> jnp.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    patterns = []

    for mask in range(2**height):
        rows = [float((mask >> row) & 1) for row in range(height)]
        patterns.append([[value] * width for value in rows])

    for mask in range(2**width):
        columns = [float((mask >> column) & 1) for column in range(width)]
        patterns.append([columns[:] for _ in range(height)])

    deduped = list(dict.fromkeys(tuple(value for row in pattern for value in row) for pattern in patterns))
    data = jnp.asarray(deduped, dtype=jnp.float32)

    if flatten:
        return data

    return data.reshape((data.shape[0], height, width))


class BarsAndStripesDataset(Dataset):
    def __init__(self, height: int = 5, width: int = 5):
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        self.height = int(height)
        self.width = int(width)
        self.n_qubits = self.height * self.width
        self.support = np.asarray(bars_and_stripes(self.height, self.width, flatten=True), dtype=np.int8)
        self.support_size = int(self.support.shape[0])
        self._support_set = {tuple(row.tolist()) for row in self.support}
        self.data = None

    def generate(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.support_size, size=int(n_samples), replace=True)
        self.data = self.support[indices].astype(np.int8)
        return self.data

    def valid_count(self, samples: np.ndarray) -> int:
        values = self._validate_samples(samples)
        return sum(tuple(row.tolist()) in self._support_set for row in values)

    def validity_rate(self, samples: np.ndarray) -> float:
        values = self._validate_samples(samples)
        if len(values) == 0:
            return 0.0
        return float(self.valid_count(values) / len(values))

    def unique_valid_count(self, samples: np.ndarray) -> int:
        values = self._validate_samples(samples)
        return len({tuple(row.tolist()) for row in values if tuple(row.tolist()) in self._support_set})

    def _validate_samples(self, samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=np.int8)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.n_qubits:
            raise ValueError(f"expected samples with shape (n, {self.n_qubits})")
        return (values > 0).astype(np.int8)


@register_builder("bars_and_stripes")
def build_bundle(config: dict, n_samples: int, seed: int) -> DatasetBundle:
    dataset = BarsAndStripesDataset(
        height=int(config.get("height", 5)),
        width=int(config.get("width", 5)),
    )
    n_qubits = int(config.get("n_qubits", dataset.n_qubits))
    if n_qubits != dataset.n_qubits:
        raise ValueError(f"bars_and_stripes n_qubits must equal height * width ({dataset.n_qubits})")
    data = dataset.generate(n_samples=n_samples, seed=seed)
    return DatasetBundle(
        dataset,
        data,
        np.asarray([], dtype=np.float64),
        dataset.n_qubits,
        "bars_and_stripes",
        EvaluationCapabilities(sampled_validity=True),
    )
