"""Rydberg atom lattice dataset.

Loads experimental measurement bitstrings from the PennyLane ``rydberggpt``
dataset, subsamples to the desired number of qubits, and constructs an
empirical probability vector with add-one (Laplace) smoothing.
"""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

from .base import BinaryDataset


class RydbergDataset(BinaryDataset):
    """Empirical distribution from Rydberg atom lattice measurements.

    The measurement bitstrings are fetched from the PennyLane ``rydberggpt``
    dataset.  For lattices larger than ``n_qubits`` atoms, only the first
    ``n_qubits`` columns (atoms) are retained.

    The target probability vector is the empirical frequency distribution
    with add-one (Laplace) smoothing to eliminate absolute zeros:

    .. math::

        p(x) = \\frac{\\text{count}(x) + 1}{N_{\\text{total}} + 2^n}

    Attributes:
        probs (np.ndarray): Smoothed empirical probability vector of shape
            ``(2**n_qubits,)`` over the computational basis.
        raw_samples (np.ndarray): The raw measurement bitstrings from the
            dataset (after subsampling to ``n_qubits`` columns).
    """

    def __init__(self, n_qubits: int = 16, dataset_index: int = 0):
        """
        Args:
            n_qubits: Number of atoms / qubits to retain.  Must be ≤ 20.
                If the loaded lattice has more atoms, the first ``n_qubits``
                columns are selected.
            dataset_index: Index of the dataset entry to load from the
                PennyLane ``rydberggpt`` collection.
        """
        super().__init__()
        if n_qubits > 20:
            raise ValueError(f"n_qubits must be ≤ 20, got {n_qubits}")
        self.n_qubits = n_qubits
        self.dataset_index = dataset_index

        self._valid_patterns = None  # Full-support distribution

        # Load data and compute empirical distribution
        self.raw_samples, self.probs = self._load_and_process()

    def _load_and_process(self):
        """Load rydberggpt data from PennyLane and compute empirical probs."""
        # Download / load from cache
        # NOTE: The rydberggpt dataset is ~22 GB.  The first download may take
        # a long time and requires sufficient disk space.
        try:
            datasets = qml.data.load(
                "other", name="rydberggpt",
                progress_bar=True, folder_path="./data",
            )
        except OSError as exc:
            raise RuntimeError(
                "Failed to load the 'rydberggpt' dataset.  This is likely "
                "due to an incomplete or corrupted download (~22 GB).  "
                "Delete any partial file in ./data/rydberggpt/ and retry "
                "with a stable connection.\n"
                f"Original error: {exc}"
            ) from exc

        if not datasets:
            raise RuntimeError(
                "No datasets returned from qml.data.load for 'rydberggpt'. "
                "Check your internet connection and PennyLane version."
            )

        ds = datasets[self.dataset_index] if self.dataset_index < len(datasets) else datasets[0]

        # Extract measurement bitstrings
        # The rydberggpt dataset stores samples; try common attribute names
        raw = None
        for attr_name in ('samples', 'bitstrings', 'measurements', 'data', 'train'):
            if hasattr(ds, attr_name):
                candidate = getattr(ds, attr_name)
                if candidate is not None:
                    raw = np.asarray(candidate)
                    if raw.ndim >= 2:
                        break

        if raw is None or raw.ndim < 2:
            # Fallback: inspect all attributes for array-like data
            for attr_name in dir(ds):
                if attr_name.startswith('_'):
                    continue
                try:
                    candidate = getattr(ds, attr_name)
                    arr = np.asarray(candidate)
                    if arr.ndim == 2 and arr.shape[0] > 10 and arr.shape[1] > 1:
                        raw = arr
                        break
                except (TypeError, ValueError):
                    continue

        if raw is None or raw.ndim < 2:
            raise RuntimeError(
                f"Could not find measurement bitstrings in rydberggpt dataset. "
                f"Available attributes: {[a for a in dir(ds) if not a.startswith('_')]}"
            )

        # Ensure binary {0, 1}
        if raw.min() < 0:
            raw = ((raw + 1) / 2).astype(np.int8)
        else:
            raw = raw.astype(np.int8)

        # Subsample to n_qubits columns
        n_atoms = raw.shape[1]
        if n_atoms < self.n_qubits:
            raise ValueError(
                f"Requested n_qubits={self.n_qubits} but dataset only has "
                f"{n_atoms} atoms. Choose a smaller n_qubits."
            )
        raw = raw[:, :self.n_qubits]

        # Compute empirical probability vector with Laplace smoothing
        N = self.n_qubits
        n_states = 2 ** N

        # Convert bitstrings to integer indices
        bit_positions = np.arange(N)
        indices = np.sum(raw.astype(int) * (2 ** bit_positions), axis=1)

        counts = np.bincount(indices, minlength=n_states).astype(np.float64)

        # Add-one (Laplace) smoothing
        smoothed_counts = counts + 1.0
        probs = smoothed_counts / smoothed_counts.sum()

        return raw, probs

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """Sample bitstrings from the smoothed empirical distribution.

        Args:
            n_samples: Number of samples to draw.
            seed: Random seed for reproducibility.

        Returns:
            Array of shape ``(n_samples, n_qubits)`` with binary values.
        """
        rng = np.random.default_rng(seed)
        p = np.asarray(self.probs, dtype=np.float64)
        p = p / p.sum()
        indices = rng.choice(len(p), size=n_samples, p=p)

        bit_positions = np.arange(self.n_qubits)
        samples = ((indices[:, None] >> bit_positions) & 1).astype(np.int8)

        self.data = samples
        return self.data

    def validity_rate(self, samples: np.ndarray) -> float:
        """Full-support (smoothed) distribution -> always 1.0."""
        return 1.0

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        """Not meaningful for smoothed empirical distributions -> always 1.0."""
        return 1.0

    def visualize(self, sample: np.ndarray, ax=None):
        """Visualize a sample as a 1D atom chain (excited / ground)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(4, self.n_qubits * 0.5), 2))

        colors = ['#F44336' if b else '#E0E0E0' for b in sample]

        # Draw atoms as circles
        for i, (c, b) in enumerate(zip(colors, sample)):
            circle = plt.Circle((i, 0), 0.35, color=c,
                                edgecolor='black', linewidth=0.5)
            ax.add_patch(circle)
            ax.text(i, 0, '|r⟩' if b else '|g⟩',
                    ha='center', va='center', fontsize=7,
                    fontweight='bold')

        ax.set_xlim(-0.5, self.n_qubits - 0.5)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.n_qubits))
        ax.set_yticks([])
        ax.set_xlabel('Atom index')
        ax.set_title(f'Rydberg lattice (N={self.n_qubits})')
        return ax
