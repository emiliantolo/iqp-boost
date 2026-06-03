"""Sampling and evaluation policy for IQP boosting experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.datasets.boltzmann_metrics import pairwise_correlation_frobenius_error
from src.core.metrics import (
    compute_jsd,
    compute_kl_divergence,
    compute_metrics,
    compute_mmd,
    compute_precision_recall_f1,
    compute_tvd,
)


def evaluate_samples(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    sigma: float | list,
    validity_fn: Callable | None = None,
    coverage_fn: Callable | None = None,
    exact_probs: np.ndarray | None = None,
    generation_eval_fn: Callable | None = None,
) -> dict:
    """Evaluate all sample-based metrics used by experiment runners."""
    mmd = compute_mmd(ground_truth, samples, sigma)

    kl = compute_kl_divergence(ground_truth, samples, exact_probs=exact_probs)
    jsd = compute_jsd(ground_truth, samples, exact_probs=exact_probs)
    tvd = compute_tvd(ground_truth, samples, exact_probs=exact_probs)
    corr_fro = pairwise_correlation_frobenius_error(ground_truth, samples, exact_probs=exact_probs)

    if validity_fn is not None and coverage_fn is not None:
        metrics = compute_metrics(ground_truth, samples, validity_fn, coverage_fn)
        sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
        prf_metrics = compute_precision_recall_f1(ground_truth, samples, sigmas[0])
    else:
        metrics = {"validity_rate": float("nan"), "coverage": float("nan")}
        prf_metrics = {
            "precision": float("nan"),
            "recall": float("nan"),
            "support_match": float("nan"),
            "f_score": float("nan"),
        }

    stats = {
        "mmd": mmd,
        "kl": kl,
        "jsd": jsd,
        "tvd": tvd,
        "validity": metrics["validity_rate"],
        "coverage": metrics["coverage"],
        "precision": prf_metrics["precision"],
        "recall": prf_metrics["recall"],
        "support_match": prf_metrics["support_match"],
        "f_score": prf_metrics["f_score"],
        "corr_fro": corr_fro,
    }
    if generation_eval_fn is not None:
        stats.update(generation_eval_fn(samples))
    return stats


def compute_ensemble_training_mmd(ensemble, ground_truth: np.ndarray) -> float:
    """Compute analytical ensemble MMD^2 wrt data using cached trace estimates."""
    if not ensemble.models or not ensemble.weights:
        return float("nan")

    n_samples = ensemble.n_samples
    m = len(ground_truth)
    n_sigmas = len(ensemble.terms.ops)
    if n_sigmas == 0:
        return float("nan")

    weights = np.asarray(ensemble.weights, dtype=float)
    mmd_vals = []

    for sigma_idx, (_, visible_ops) in ensemble.terms.ops.items():
        tr_data = np.mean(1 - 2 * ((ground_truth @ np.asarray(visible_ops).T) % 2), axis=0)

        tr_enss = np.asarray([np.asarray(t[sigma_idx]) for t in ensemble.terms.trs])
        corr_enss = np.asarray([np.asarray(c[sigma_idx]) for c in ensemble.terms.corrs])

        tr_mix = np.sum(weights[:, None] * tr_enss, axis=0)
        tr_mix_sq = np.einsum("i,ik,jk,j->k", weights, tr_enss, tr_enss, weights)
        corr_mix = np.sum((weights**2)[:, None] * corr_enss, axis=0)

        term_mix_mix = np.mean((tr_mix_sq - corr_mix) * n_samples / (n_samples - 1))
        term_mix_data = np.mean(tr_mix * tr_data)
        term_data_data = np.mean((tr_data * tr_data * m - 1) / (m - 1))

        mmd_vals.append(term_mix_mix - 2.0 * term_mix_data + term_data_data)

    return float(np.mean(mmd_vals))


@dataclass(frozen=True)
class EvaluationPolicy:
    """Owns sampling flags, seeds, and metric context for an experiment run."""

    x_train: np.ndarray
    sigma: float | list
    shots: int
    rng_seed: int
    skip_sampling: bool = False
    final_eval_sampling: bool = False
    validity_fn: Callable | None = None
    coverage_fn: Callable | None = None
    exact_probs: np.ndarray | None = None
    generation_eval_fn: Callable | None = None

    @property
    def sampling_enabled(self) -> bool:
        return not self.skip_sampling

    @property
    def final_sampling_enabled(self) -> bool:
        return (not self.skip_sampling) or self.final_eval_sampling

    def rng_for_step(self, step: int) -> np.random.Generator:
        return np.random.default_rng(self.rng_seed + step * 7919)

    def evaluate_samples(self, samples: np.ndarray) -> dict:
        return evaluate_samples(
            self.x_train,
            samples,
            self.sigma,
            self.validity_fn,
            self.coverage_fn,
            exact_probs=self.exact_probs,
            generation_eval_fn=self.generation_eval_fn,
        )

    def evaluate_ensemble_training_mmd(self, ensemble, ground_truth: np.ndarray | None = None) -> float:
        return compute_ensemble_training_mmd(
            ensemble,
            self.x_train if ground_truth is None else ground_truth,
        )

    def sample_and_evaluate_ensemble(self, ensemble, step: int) -> tuple[np.ndarray, dict]:
        samples = ensemble.sample(self.shots, self.rng_for_step(step))
        return samples, self.evaluate_samples(samples)

    def sample_and_evaluate_circuit(self, circuit, params, wires: list | None = None) -> tuple[np.ndarray, dict]:
        try:
            samples = circuit.sample(params, shots=self.shots, wires=wires)
        except TypeError:
            samples = circuit.sample(params, shots=self.shots)
        if wires is not None and samples.shape[1] != len(wires):
            samples = samples[:, wires]
        return samples, self.evaluate_samples(samples)

    def analytical_mmd_stats(self, value: float) -> dict:
        return {"mmd": value}
