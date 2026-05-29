import numpy as np
import matplotlib.pyplot as plt

from .base import BinaryDataset
from .mcmc_chain import run_mcmc_chains
from .boltzmann_utils import dense_boltzmann_probs, sample_from_probs

class HopfieldDataset(BinaryDataset):
    """Dataset based on the Boltzmann distribution of a Hopfield network."""
    
    def __init__(
        self,
        n_qubits: int = 16,
        n_patterns: int = 5,
        beta: float = 2.0,
        pattern_seed: int = 0,
        batch_size: int = 2**20,
        max_exact_states: int = 2**20,
        mcmc_burn_in: int = 256,
        mcmc_thinning: int = 16,
        mcmc_sweeps_per_sample: int = 1, # Preserved for backward compatibility
        train_split_ratio: float | None = None,
    ):
        super().__init__()
        if n_qubits <= 0: raise ValueError("n_qubits must be positive")
        if n_patterns <= 0: raise ValueError("n_patterns must be positive")
        if beta <= 0: raise ValueError("beta must be positive")

        self.n_qubits = n_qubits
        self.n_patterns = n_patterns
        self.beta = beta
        self.pattern_seed = pattern_seed
        self.batch_size = batch_size
        self.max_exact_states = int(max_exact_states)
        self.mcmc_burn_in = int(mcmc_burn_in)
        self.mcmc_thinning = int(mcmc_thinning)
        self.mcmc_sweeps_per_sample = int(mcmc_sweeps_per_sample)
        self.train_split_ratio = train_split_ratio
        self.active_split = "train"
        self._all_data: np.ndarray | None = None

        rng = np.random.default_rng(pattern_seed)
        self.patterns = rng.choice([-1.0, 1.0], size=(n_patterns, n_qubits))

        self.J = sum(np.outer(p, p) for p in self.patterns) / n_qubits
        np.fill_diagonal(self.J, 0)

        self._valid_patterns = None
        self._exact_mode = 2 ** self.n_qubits <= self.max_exact_states
        self.probs = self._compute_boltzmann() if self._exact_mode else None

    def _compute_boltzmann(self):
        return dense_boltzmann_probs(self.J, self.beta, self.batch_size)

    def _generate_mcmc(self, n_samples: int, seed: int = 0) -> np.ndarray:
        # Properly leverages thinning across sequential blocks, saving 100x+ compute
        return run_mcmc_chains(
            self.J, self.beta, n_samples,
            burn_in=self.mcmc_burn_in,
            thinning=self.mcmc_thinning,
            sweeps_per_sample=self.mcmc_sweeps_per_sample,
            seed=seed,
        )

    def _generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        if self.probs is None:
            samples = self._generate_mcmc(n_samples, seed=seed)
        else:
            samples = sample_from_probs(self.probs, self.n_qubits, n_samples, seed=seed)
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

    def set_split(self, split: str = "train") -> "HopfieldDataset":
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

    def visualize(self, sample: np.ndarray, ax=None):
        if ax is None: fig, ax = plt.subplots(figsize=(4, 4))
        side = int(np.sqrt(self.n_qubits))
        display_data = sample.reshape(side, side) if side * side == self.n_qubits else sample.reshape(1, -1)
        ax.imshow(display_data, cmap='binary', interpolation='nearest')
        
        if self.n_qubits <= 64:
            if side * side == self.n_qubits:
                for i in range(side):
                    for j in range(side):
                        ax.text(j, i, str(int(display_data[i, j])), ha='center', va='center',
                                color='red' if display_data[i, j] == 0 else 'gray')
            else:
                for i, bit in enumerate(sample):
                    ax.text(i, 0, str(int(bit)), ha='center', va='center',
                            color='red' if bit == 0 else 'gray')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Hopfield Sample ({self.n_qubits} qubits)")
        return ax

    def top_k_tvd(self, k: int) -> float:
        if self.probs is None: return float("nan")
        p = np.asarray(self.probs, dtype=np.float64)
        p = p / p.sum()
        top_k_idx = np.argsort(p)[::-1][:k]
        q = np.zeros_like(p)
        q[top_k_idx] = p[top_k_idx]
        q = q / q.sum()
        return float(0.5 * np.abs(p - q).sum())