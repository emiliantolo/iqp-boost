import numpy as np
import matplotlib.pyplot as plt

from .base import BinaryDataset
from .boltzmann_utils import dense_boltzmann_probs, sample_from_probs
from .mcmc_chain import run_mcmc_chains


class FrustratedIsingDataset(BinaryDataset):
    """Samples from a Boltzmann distribution on either a 2D grid or an SK model."""

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        beta: float = 1.0,
        j_seed: int = 0,
        batch_size: int = 2**20,
        max_exact_states: int = 2**20,
        model: str = 'grid',
        mcmc_burn_in: int = 256,
        mcmc_thinning: int = 4,
        mcmc_sweeps_per_sample: int = 1,
        train_split_ratio: float | None = None,
    ):
        super().__init__()
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if model not in {'grid', 'sk'}:
            raise ValueError("model must be 'grid' or 'sk'")

        self.rows = int(rows)
        self.cols = int(cols)
        self.n_qubits = self.rows * self.cols
        self.beta = float(beta)
        self.j_seed = int(j_seed)
        self.batch_size = int(batch_size)
        self.max_exact_states = int(max_exact_states)
        self.model = model
        self.mcmc_burn_in = int(mcmc_burn_in)
        self.mcmc_thinning = int(mcmc_thinning)
        self.mcmc_sweeps_per_sample = int(mcmc_sweeps_per_sample)
        self.train_split_ratio = train_split_ratio
        self.active_split = "train"
        self._all_data: np.ndarray | None = None

        self.J_dense, self.edges, self.J_weights = self._build_couplings()
        self._valid_patterns = None
        self._exact_mode = 2 ** self.n_qubits <= self.max_exact_states
        self.probs = dense_boltzmann_probs(self.J_dense, self.beta, self.batch_size) if self._exact_mode else None

    def _build_couplings(self):
        if self.model == 'sk':
            rng = np.random.default_rng(self.j_seed)
            J_dense = rng.normal(loc=0.0, scale=1.0 / np.sqrt(self.n_qubits), size=(self.n_qubits, self.n_qubits))
            J_dense = 0.5 * (J_dense + J_dense.T)
            np.fill_diagonal(J_dense, 0.0)
            return J_dense, None, None

        edges = []
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                if c + 1 < self.cols:
                    edges.append([idx, idx + 1])
                if r + 1 < self.rows:
                    edges.append([idx, idx + self.cols])

        rng = np.random.default_rng(self.j_seed)
        J_weights = rng.choice([-1.0, 1.0], size=len(edges))
        J_dense = np.zeros((self.n_qubits, self.n_qubits), dtype=np.float64)
        for (u, v), w in zip(edges, J_weights):
            J_dense[u, v] = w
            J_dense[v, u] = w
        return J_dense, np.asarray(edges, dtype=np.int32), np.asarray(J_weights, dtype=np.float64)

    def _generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        if self.probs is not None:
            samples = sample_from_probs(self.probs, self.n_qubits, n_samples, seed=seed)
        else:
            samples = run_mcmc_chains(
                self.J_dense,
                self.beta,
                n_samples,
                burn_in=self.mcmc_burn_in,
                thinning=self.mcmc_thinning,
                sweeps_per_sample=self.mcmc_sweeps_per_sample,
                seed=seed,
            )
        return samples

    def generate(self, n_samples: int | None = None, seed: int = 0, split: str = "train") -> np.ndarray:
        if self.train_split_ratio is None:
            samples = self._generate(n_samples, seed=seed)
            self.data = samples
            return self.data

        # Split mode: generate once, cache, then return the requested split
        if self._all_data is None:
            total = n_samples
            if total is None:
                raise ValueError("n_samples must be provided when train_split_ratio is set")
            self._all_data = self._generate(total, seed=seed)
            split_idx = int(len(self._all_data) * self.train_split_ratio)
            if split_idx == 0 or split_idx == len(self._all_data):
                raise ValueError("train_split_ratio produced an empty train or test split")
            self._train_data = self._all_data[:split_idx]
            self._test_data = self._all_data[split_idx:]

        if split == "train":
            self.data = self._train_data
        elif split == "test":
            self.data = self._test_data
        elif split == "all":
            self.data = self._all_data
        else:
            raise ValueError(f"split must be 'train', 'test', or 'all', got {split!r}")
        return self.data

    def set_split(self, split: str = "train") -> "FrustratedIsingDataset":
        if split in {"train", "test", "all"}:
            self.active_split = split
        else:
            raise ValueError(f"split must be 'train', 'test', or 'all', got {split!r}")
        return self

    def validity_rate(self, samples: np.ndarray) -> float:
        """Full-support Boltzmann distributions make strict validity trivial."""
        return 1.0

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        """Full-support Boltzmann distributions make strict coverage trivial."""
        return 1.0

    def top_k_tvd(self, k: int) -> float:
        if self.probs is None:
            return float("nan")
        p = np.asarray(self.probs, dtype=np.float64)
        p = p / p.sum()
        top_k_idx = np.argsort(p)[::-1][:k]
        q = np.zeros_like(p)
        q[top_k_idx] = p[top_k_idx]
        q = q / q.sum()
        return float(0.5 * np.abs(p - q).sum())

    def visualize(self, sample: np.ndarray, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(max(3, self.cols * 0.7), max(3, self.rows * 0.7)))

        spins = 1 - 2 * sample.reshape(self.rows, self.cols).astype(float)
        ax.imshow(spins, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
        for r in range(self.rows):
            for c in range(self.cols):
                ax.text(
                    c,
                    r,
                    '+' if spins[r, c] > 0 else '-',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='white' if spins[r, c] > 0 else 'black',
                    fontweight='bold',
                )
        for i in range(self.rows + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.cols + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Ising {self.model.upper()} {self.rows}x{self.cols}  beta={self.beta}')
        return ax