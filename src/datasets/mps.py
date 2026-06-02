import numpy as np
import matplotlib.pyplot as plt

from .base import BinaryDataset
from .boltzmann_utils import sample_from_probs


class MPSDataset(BinaryDataset):
    """Matrix Product States benchmark: 1D tensor network with low bond dimension.

    Probability (Born rule): P(x) = |Tr(A_1^{x_1} ... A_n^{x_n})|^2 / Z

    Exact mode (n <= 20): enumerates all 2^n bitstrings.
    Scalable mode (n > 20): left-to-right ancestral sampling using the
    probability MPO (bond dimension = chi^2).
    """

    def __init__(
        self,
        n_qubits: int = 16,
        chi: int = 4,
        seed: int = 42,
        batch_size: int = 2**20,
        max_exact_states: int = 2**20,
    ):
        super().__init__()
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if chi <= 0:
            raise ValueError("chi must be positive")

        self.n_qubits = int(n_qubits)
        self.chi = int(chi)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.max_exact_states = int(max_exact_states)
        self._valid_patterns = None

        self.tensors = self._init_tensors()
        self._build_mpo()
        self._build_right_envs()

        self._exact_mode = 2**self.n_qubits <= self.max_exact_states
        self.probs = self._compute_exact_probs() if self._exact_mode else None

    def _init_tensors(self) -> list[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        tensors = []
        chi = self.chi
        for _ in range(self.n_qubits):
            A = rng.normal(0, 1, size=(2, chi, chi)).astype(np.float64)
            for b in (0, 1):
                scale = np.linalg.norm(A[b], 'fro')
                if scale > 0:
                    A[b] /= scale
            tensors.append(A)
        return tensors

    def _build_mpo(self):
        chi = self.chi
        self.mpo = []
        for i in range(self.n_qubits):
            A = self.tensors[i]
            M = np.zeros((2, chi * chi, chi * chi), dtype=np.float64)
            for b in (0, 1):
                M[b] = np.kron(A[b], A[b])
            self.mpo.append(M)

    def _build_right_envs(self):
        chi2 = self.chi * self.chi
        T = [self.mpo[j][0] + self.mpo[j][1] for j in range(self.n_qubits)]
        self._R = [None] * self.n_qubits
        rv = np.eye(chi2, dtype=np.float64)
        for j in range(self.n_qubits - 1, -1, -1):
            rv = T[j] @ rv
            self._R[j] = rv

    def _contract_all(self, x: np.ndarray) -> float:
        mat = self.tensors[0][int(x[0])]
        for i in range(1, self.n_qubits):
            mat = mat @ self.tensors[i][int(x[i])]
        return np.trace(mat)

    def _compute_exact_probs(self) -> np.ndarray:
        total = 2**self.n_qubits
        indices = np.arange(total, dtype=np.int64)[:, None]
        bit_shifts = np.arange(self.n_qubits, dtype=np.int64)
        all_bits = ((indices >> bit_shifts) & 1).astype(np.int8)
        unnorm = np.zeros(total, dtype=np.float64)
        for idx in range(total):
            val = self._contract_all(all_bits[idx])
            unnorm[idx] = val * val
        z = unnorm.sum()
        return unnorm / z if z > 0 else np.ones(total) / total

    def probability(self, x: np.ndarray) -> float:
        val = self._contract_all(np.asarray(x, dtype=np.int8))
        unnorm_p = val * val
        Z = np.trace(self._R[0])
        p = unnorm_p / Z if Z > 0 else unnorm_p
        return p if p > 0 else 1e-30

    def _ancestral_sample(self, n_samples: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = self.n_qubits
        chi2 = self.chi * self.chi
        R = self._R
        samples = np.empty((n_samples, n), dtype=np.int8)

        for s in range(n_samples):
            P = np.eye(chi2, dtype=np.float64)
            for i in range(n):
                M0, M1 = self.mpo[i][0], self.mpo[i][1]
                R_next = R[i + 1] if i < n - 1 else np.eye(chi2, dtype=np.float64)

                TMP0 = P @ M0
                TMP1 = P @ M1
                s0 = np.trace(TMP0 @ R_next)
                s1 = np.trace(TMP1 @ R_next)

                total = s0 + s1
                prob1 = s1 / total if total > 0 else 0.5

                bit = 1 if rng.random() < prob1 else 0
                samples[s, i] = np.int8(bit)

                P = P @ self.mpo[i][bit]
                norm = np.linalg.norm(P)
                if norm > 0:
                    P /= norm

        return samples

    def generate(self, n_samples: int | None = None, seed: int = 0, split: str = "train") -> np.ndarray:
        if self.probs is not None:
            samples = sample_from_probs(self.probs, self.n_qubits, n_samples, seed=seed)
        else:
            samples = self._ancestral_sample(n_samples, seed=seed)
        self.data = samples
        return self.data

    def validity_rate(self, samples: np.ndarray) -> float:
        return 1.0

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        return 1.0

    def visualize(self, sample: np.ndarray, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(max(3, self.n_qubits * 0.15), 2))
        ax.imshow(sample.reshape(1, -1), cmap='binary', interpolation='nearest', aspect='auto')
        ax.set_yticks([])
        ax.set_xticks(range(self.n_qubits))
        ax.set_xticklabels([])
        ax.set_title(f'MPS (n={self.n_qubits}, chi={self.chi})')
        return ax
