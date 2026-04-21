"""Random Quantum Volume / random circuit dataset.

Generates the output probability distribution of a Quantum Volume-style
random circuit using PennyLane statevector simulation.  The circuit uses
``depth`` layers of random 2-qubit Haar unitaries on random qubit pairings,
producing a Porter-Thomas speckle pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from scipy.stats import unitary_group

from .base import BinaryDataset


class RandomCircuitDataset(BinaryDataset):
    """Output distribution of a random Quantum Volume-style circuit.

    The circuit consists of ``depth`` layers.  In each layer the qubits are
    randomly paired and a Haar-random SU(4) unitary is applied to each pair.
    The exact computational-basis probabilities are obtained via statevector
    simulation.

    Attributes:
        probs (np.ndarray): Exact probability vector of shape
            ``(2**n_qubits,)`` over the computational basis.
    """

    def __init__(self, n_qubits: int = 12, depth: int | None = None,
                 circuit_seed: int = 42):
        """
        Args:
            n_qubits: Number of qubits. Recommended: 8, 12, 16, 20.
            depth: Number of random circuit layers.  Defaults to ``n_qubits``.
            circuit_seed: Seed for generating the random unitaries and pairings.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth if depth is not None else n_qubits
        self.circuit_seed = circuit_seed

        self._valid_patterns = None  # Full-support distribution

        # Build and simulate the random circuit (cached)
        self.probs = self._load_or_compute()

    def _cache_key(self) -> str:
        import hashlib
        params = (self.n_qubits, self.depth, self.circuit_seed)
        h = hashlib.sha256(repr(params).encode()).hexdigest()[:16]
        return f"rc_N{self.n_qubits}_d{self.depth}_s{self.circuit_seed}_{h}"

    def _load_or_compute(self) -> np.ndarray:
        """Load cached probs or simulate and cache."""
        from pathlib import Path
        cache_dir = Path("./data/cache/random_circuit")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self._cache_key()}.npz"

        if cache_file.exists():
            data = np.load(cache_file)
            print(f"[RandomCircuitDataset] Loaded from cache: {cache_file}")
            return data['probs']

        probs = self._simulate_random_circuit()
        np.savez(cache_file, probs=probs)
        print(f"[RandomCircuitDataset] Saved to cache: {cache_file}")
        return probs

    def _simulate_random_circuit(self) -> np.ndarray:
        """Build a QV-style random circuit and return exact output probabilities."""
        rng = np.random.default_rng(self.circuit_seed)
        N = self.n_qubits
        d = self.depth

        # Pre-generate all random unitaries and permutations
        n_pairs = N // 2
        permutations = []
        unitaries = []
        for _ in range(d):
            perm = rng.permutation(N).tolist()
            permutations.append(perm)
            layer_unitaries = []
            for _ in range(n_pairs):
                U = unitary_group.rvs(4, random_state=rng)
                layer_unitaries.append(U)
            unitaries.append(layer_unitaries)

        dev = qml.device("default.qubit", wires=N)

        @qml.qnode(dev)
        def circuit():
            # Initial Hadamard layer for superposition
            for q in range(N):
                qml.Hadamard(wires=q)

            for layer_idx in range(d):
                perm = permutations[layer_idx]
                for pair_idx in range(n_pairs):
                    q0 = perm[2 * pair_idx]
                    q1 = perm[2 * pair_idx + 1]
                    U = unitaries[layer_idx][pair_idx]
                    qml.QubitUnitary(U, wires=[q0, q1])

            return qml.probs(wires=range(N))

        probs = np.asarray(circuit(), dtype=np.float64)
        return probs

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """Sample bitstrings from the exact random circuit distribution.

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
        """Full-support distribution -> always 1.0."""
        return 1.0

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        """Not meaningful for full-support distributions -> always 1.0."""
        return 1.0

    def visualize(self, sample: np.ndarray, ax=None):
        """Visualize a sample as a bitstring bar chart."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(4, self.n_qubits * 0.4), 2))

        ax.bar(range(self.n_qubits), sample.astype(float),
               color=['#2196F3' if b else '#E0E0E0' for b in sample],
               edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(self.n_qubits))
        ax.set_yticks([0, 1])
        ax.set_xlabel('Qubit')
        ax.set_title(f'Random Circuit (N={self.n_qubits}, d={self.depth})')
        return ax
