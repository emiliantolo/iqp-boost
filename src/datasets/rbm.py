from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from .base import BinaryDataset
from .boltzmann_utils import sample_from_probs


class RBMDataset(BinaryDataset):
    """Restricted Boltzmann Machine target distribution on visible units only.

    The distribution is defined by a randomly initialized bipartite energy
    model with binary visible and hidden units:

        E(v, h) = -b^T v - c^T h - v^T W h

    Exact mode marginalizes out h analytically and returns the visible-layer
    distribution. MCMC mode alternates Gibbs updates on h and v.
    """

    def __init__(
        self,
        n_visible: int = 16,
        n_hidden: int | None = None,
        beta: float = 1.0,
        weight_seed: int = 0,
        bias_seed: int | None = None,
        weight_scale: float = 1.0,
        visible_bias_scale: float = 0.25,
        hidden_bias_scale: float = 0.25,
        batch_size: int = 2**20,
        max_exact_states: int = 2**20,
        mcmc_burn_in: int = 256,
        mcmc_thinning: int = 4,
        mcmc_sweeps_per_sample: int = 1,
    ):
        super().__init__()
        if n_visible <= 0:
            raise ValueError("n_visible must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if n_hidden is None:
            n_hidden = max(1, n_visible // 2)
        if n_hidden <= 0:
            raise ValueError("n_hidden must be positive")

        self.n_visible = int(n_visible)
        self.n_hidden = int(n_hidden)
        self.n_qubits = self.n_visible
        self.beta = float(beta)
        self.weight_seed = int(weight_seed)
        self.bias_seed = int(bias_seed if bias_seed is not None else weight_seed + 1)
        self.weight_scale = float(weight_scale)
        self.visible_bias_scale = float(visible_bias_scale)
        self.hidden_bias_scale = float(hidden_bias_scale)
        self.batch_size = int(batch_size)
        self.max_exact_states = int(max_exact_states)
        self.mcmc_burn_in = int(mcmc_burn_in)
        self.mcmc_thinning = int(mcmc_thinning)
        self.mcmc_sweeps_per_sample = int(mcmc_sweeps_per_sample)

        rng_w = np.random.default_rng(self.weight_seed)
        rng_b = np.random.default_rng(self.bias_seed)
        scale = self.weight_scale / np.sqrt(self.n_visible)
        self.W = rng_w.normal(loc=0.0, scale=scale, size=(self.n_visible, self.n_hidden))
        self.visible_bias = rng_b.normal(loc=0.0, scale=self.visible_bias_scale, size=self.n_visible)
        self.hidden_bias = rng_b.normal(loc=0.0, scale=self.hidden_bias_scale, size=self.n_hidden)

        self._exact_mode = 2 ** self.n_visible <= self.max_exact_states
        self.probs = self._compute_visible_probs() if self._exact_mode else None
        self._valid_patterns = None

    def _visible_log_unnormalized(self, visible_bits: np.ndarray) -> np.ndarray:
        v = np.asarray(visible_bits, dtype=np.float64)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        hidden_logits = self.beta * (v @ self.W + self.hidden_bias)
        visible_term = self.beta * (v @ self.visible_bias)
        hidden_term = np.logaddexp(0.0, hidden_logits).sum(axis=1)
        return visible_term + hidden_term

    def _compute_visible_probs(self) -> np.ndarray:
        total_states = 2 ** self.n_visible
        bs = min(self.batch_size, total_states)

        log_weights = []
        for start in range(0, total_states, bs):
            stop = min(start + bs, total_states)
            idx = np.arange(start, stop, dtype=np.int64)
            bits = ((idx[:, None] >> np.arange(self.n_visible)) & 1).astype(np.float64)
            log_weights.append(self._visible_log_unnormalized(bits))

        log_weights = np.concatenate(log_weights)
        log_z = logsumexp(log_weights)
        return np.exp(log_weights - log_z)

    def _sample_hidden_given_visible(self, visible: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        logits = self.beta * (visible @ self.W + self.hidden_bias)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (rng.random(probs.shape) < probs).astype(np.int8)

    def _sample_visible_given_hidden(self, hidden: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        logits = self.beta * (hidden @ self.W.T + self.visible_bias)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (rng.random(probs.shape) < probs).astype(np.int8)

    def _generate_mcmc(self, n_samples: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        visible = rng.integers(0, 2, size=self.n_visible, dtype=np.int8)
        hidden = rng.integers(0, 2, size=self.n_hidden, dtype=np.int8)

        for _ in range(self.mcmc_burn_in):
            hidden = self._sample_hidden_given_visible(visible, rng)
            visible = self._sample_visible_given_hidden(hidden, rng)

        samples = np.empty((n_samples, self.n_visible), dtype=np.int8)
        for i in range(n_samples):
            for _ in range(self.mcmc_thinning):
                for _ in range(self.mcmc_sweeps_per_sample):
                    hidden = self._sample_hidden_given_visible(visible, rng)
                    visible = self._sample_visible_given_hidden(hidden, rng)
            samples[i] = visible

        return samples

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        if self.probs is not None:
            samples = sample_from_probs(self.probs, self.n_visible, n_samples, seed=seed)
        else:
            samples = self._generate_mcmc(n_samples, seed=seed)
        self.data = samples
        return self.data

    def validity_rate(self, samples: np.ndarray) -> float:
        """Full-support RBM distributions make strict validity trivial."""
        return 1.0

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        """Full-support RBM distributions make strict coverage trivial."""
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
            _, ax = plt.subplots(figsize=(max(4, self.n_visible * 0.45), 2.5))

        sample = np.asarray(sample, dtype=np.int8)
        side = int(np.sqrt(self.n_visible))
        if side * side == self.n_visible:
            display_data = sample.reshape(side, side)
        else:
            display_data = sample.reshape(1, -1)

        ax.imshow(display_data, cmap='binary', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'RBM visible layer ({self.n_visible}v, {self.n_hidden}h)')
        return ax
