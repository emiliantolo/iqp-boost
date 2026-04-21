"""Thermal state of the 1D Transverse-Field Ising Model (TFIM).

Computes the exact Gibbs state diagonal in the computational basis for the
1D TFIM Hamiltonian:

.. math::

    H = -J \\sum_i Z_i Z_{i+1} - h \\sum_i X_i

at a configurable finite temperature :math:`T`.  The target probabilities
are the diagonal elements of the density matrix
:math:`\\rho = e^{-H/T} / \\mathrm{Tr}(e^{-H/T})`.

Exact diagonalisation is used, which limits this dataset to **N ≤ 14** qubits.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from .base import BinaryDataset


class TFIMThermalDataset(BinaryDataset):
    """Thermal (Gibbs) state of a 1D Transverse-Field Ising Model.

    The Hamiltonian is built with ``qml.spin.transverse_ising`` and
    diagonalised exactly to produce the computational-basis Gibbs
    probabilities at temperature *T*.

    Attributes:
        probs (np.ndarray): Exact Gibbs probability vector of shape
            ``(2**n_qubits,)`` over the computational basis.
        eigenvalues (np.ndarray): Energy eigenvalues of the Hamiltonian.
    """

    MAX_QUBITS = 14  # Hard cap for exact diagonalisation

    def __init__(self, n_qubits: int = 12,
                 temperature: float = 1.0,
                 coupling: float = 1.0,
                 h_field: float = 1.0,
                 boundary_condition: bool = False):
        """
        Args:
            n_qubits: Number of spins in the 1D chain.  Must be ≤ 14.
            temperature: Gibbs temperature *T*.  Higher *T* → flatter
                distribution.  Must be > 0.
            coupling: Nearest-neighbour ZZ coupling *J*.
            h_field: Transverse magnetic field strength *h*.
            boundary_condition: If ``True``, periodic boundary conditions
                (ring topology).
        """
        super().__init__()
        if n_qubits > self.MAX_QUBITS:
            raise ValueError(
                f"n_qubits must be ≤ {self.MAX_QUBITS} for exact "
                f"diagonalisation, got {n_qubits}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.n_qubits = n_qubits
        self.temperature = temperature
        self.coupling = coupling
        self.h_field = h_field
        self.boundary_condition = boundary_condition

        self._valid_patterns = None  # Full-support distribution

        # Build Hamiltonian, diagonalise, compute Gibbs probabilities (cached)
        self.eigenvalues, self.probs = self._load_or_compute()

    def _cache_key(self) -> str:
        """Deterministic hash of all parameters that affect the result."""
        import hashlib
        params = (self.n_qubits, self.temperature, self.coupling,
                  self.h_field, self.boundary_condition)
        h = hashlib.sha256(repr(params).encode()).hexdigest()[:16]
        return f"tfim_N{self.n_qubits}_T{self.temperature}_{h}"

    def _load_or_compute(self):
        """Load cached eigenvalues/probs or compute and cache them."""
        from pathlib import Path
        cache_dir = Path("./data/cache/tfim")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self._cache_key()}.npz"

        if cache_file.exists():
            data = np.load(cache_file)
            print(f"[TFIMThermalDataset] Loaded from cache: {cache_file}")
            return data['eigenvalues'], data['probs']

        eigenvalues, probs = self._compute_gibbs_probs()
        np.savez(cache_file, eigenvalues=eigenvalues, probs=probs)
        print(f"[TFIMThermalDataset] Saved to cache: {cache_file}")
        return eigenvalues, probs

    def _compute_gibbs_probs(self):
        """Build the TFIM Hamiltonian, diagonalise, and compute Gibbs diagonal.

        Constructs the sparse Hamiltonian directly (much faster than
        ``qml.matrix`` for N > 10) and converts to dense for ``eigh``.
        """
        from scipy import sparse

        N = self.n_qubits
        T = self.temperature
        dim = 2 ** N

        # --- Build sparse TFIM Hamiltonian ---
        # H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
        #
        # ZZ terms are diagonal: Z_i Z_{i+1} |x> = (-1)^{x_i + x_{i+1}} |x>
        # X_i terms flip bit i:  X_i |...b_i...> = |...1-b_i...>

        all_states = np.arange(dim, dtype=np.int64)

        # Diagonal: -J * sum_i (1 - 2*bit_i)(1 - 2*bit_{i+1})
        diag = np.zeros(dim, dtype=np.float64)
        n_bonds = N - 1 + (1 if self.boundary_condition else 0)
        for i in range(n_bonds):
            j = (i + 1) % N
            bit_i = (all_states >> i) & 1
            bit_j = (all_states >> j) & 1
            spin_i = 1 - 2 * bit_i  # 0->+1, 1->-1
            spin_j = 1 - 2 * bit_j
            diag -= self.coupling * spin_i * spin_j

        H_sparse = sparse.diags(diag, format='csr')

        # Off-diagonal: -h * sum_i X_i  (each X_i flips bit i)
        if self.h_field != 0.0:
            for i in range(N):
                flipped = all_states ^ (1 << i)  # flip bit i
                data = np.full(dim, -self.h_field, dtype=np.float64)
                X_i = sparse.csr_matrix(
                    (data, (all_states, flipped)), shape=(dim, dim),
                )
                H_sparse = H_sparse + X_i

        # Convert to dense for exact diagonalisation
        H_dense = H_sparse.toarray()

        # Exact diagonalisation
        eigenvalues, eigenvectors = eigh(H_dense)

        # Gibbs probabilities in the computational basis:
        # p(x) = sum_k |U_{x,k}|^2 * exp(-E_k / T) / Z
        boltzmann_weights = np.exp(-eigenvalues / T)
        Z = boltzmann_weights.sum()

        # |U_{x,k}|^2 * w_k summed over k, for each x
        # eigenvectors has shape (2^N, 2^N) where column k is eigenvector k
        probs = np.einsum('xk,k->x',
                          np.abs(eigenvectors) ** 2,
                          boltzmann_weights) / Z

        return eigenvalues, probs

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """Sample bitstrings from the exact Gibbs distribution.

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
        """Visualize a sample as a 1D spin chain."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(4, self.n_qubits * 0.5), 2))

        spins = 1 - 2 * sample.astype(float)  # Map 0->+1, 1->-1
        colors = ['#F44336' if s < 0 else '#2196F3' for s in spins]

        ax.bar(range(self.n_qubits), np.ones(self.n_qubits),
               color=colors, edgecolor='black', linewidth=0.5)

        for i, s in enumerate(spins):
            ax.text(i, 0.5, '+' if s > 0 else '−',
                    ha='center', va='center', fontsize=12,
                    fontweight='bold', color='white')

        ax.set_xlim(-0.5, self.n_qubits - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(self.n_qubits))
        ax.set_yticks([])
        ax.set_xlabel('Site')
        ax.set_title(f'TFIM chain (N={self.n_qubits}, T={self.temperature}, '
                     f'J={self.coupling}, h={self.h_field})')
        return ax
