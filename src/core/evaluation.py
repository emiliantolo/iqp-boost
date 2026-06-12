"""Sampling and evaluation policy for IQP boosting experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import warnings

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
    sigma: float | list | dict,
    validity_fn: Callable | None = None,
    coverage_fn: Callable | None = None,
    exact_probs: np.ndarray | None = None,
    generation_eval_fn: Callable | None = None,
    model_probs: np.ndarray | None = None,
) -> dict:
    """Evaluate all sample-based metrics used by experiment runners."""
    mmd = compute_mmd(ground_truth, samples, sigma)

    kl = compute_kl_divergence(ground_truth, samples, exact_probs=exact_probs)
    jsd = compute_jsd(ground_truth, samples, exact_probs=exact_probs)
    tvd = compute_tvd(ground_truth, samples, exact_probs=exact_probs)
    corr_fro = pairwise_correlation_frobenius_error(ground_truth, samples, exact_probs=exact_probs)

    if validity_fn is not None and coverage_fn is not None:
        metrics = compute_metrics(ground_truth, samples, validity_fn, coverage_fn)
        if isinstance(sigma, dict):
            prf_sigma = sigma['sigma'] if isinstance(sigma['sigma'], (int, float)) else sigma['sigma'][0]
        else:
            sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
            prf_sigma = sigmas[0]
        prf_metrics = compute_precision_recall_f1(ground_truth, samples, prf_sigma)
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
    if model_probs is not None:
        stats.update(compute_exact_distribution_metrics(ground_truth, exact_probs, model_probs))
    return stats


def compute_exact_distribution_metrics(
    ground_truth: np.ndarray,
    exact_probs: np.ndarray | None,
    model_probs: np.ndarray,
) -> dict:
    """Compute exact model-distribution metrics with suffixed keys."""
    if exact_probs is not None and len(exact_probs) != len(model_probs):
        _warn_exact(
            f"exact reference length {len(exact_probs)} does not match model probability length {len(model_probs)}"
        )
        return _nan_exact_metrics()
    if not np.all(np.isfinite(model_probs)) or float(np.asarray(model_probs).sum()) <= 0:
        _warn_exact("model probabilities are non-finite or sum to a non-positive value")
        return _nan_exact_metrics()
    return {
        "kl_exact": compute_kl_divergence(ground_truth, np.empty((0, 0), dtype=np.int8), exact_probs=exact_probs, model_probs=model_probs),
        "jsd_exact": compute_jsd(ground_truth, np.empty((0, 0), dtype=np.int8), exact_probs=exact_probs, model_probs=model_probs),
        "tvd_exact": compute_tvd(ground_truth, np.empty((0, 0), dtype=np.int8), exact_probs=exact_probs, model_probs=model_probs),
    }


def _nan_exact_metrics() -> dict:
    return {"kl_exact": float("nan"), "jsd_exact": float("nan"), "tvd_exact": float("nan")}


def _default_exact_metrics_config() -> dict:
    return {
        "enabled": False,
        "max_qubits": 20,
        "phases": {
            "baseline": True,
            "steps": False,
            "final": True,
            "per_model": True,
            "fcfw": True,
        },
    }


def resolve_exact_metrics_config(config: dict | None) -> dict:
    """Normalize exact metric config, including the legacy exact_sampling alias."""
    resolved = _default_exact_metrics_config()
    config = config or {}
    if any(key in config for key in ("enabled", "max_qubits", "phases")):
        raw = dict(config)
    else:
        raw = config.get("exact_metrics", {})
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    elif raw is None:
        raw = {}

    if "exact_sampling" in config and "enabled" not in raw:
        raw = {**raw, "enabled": bool(config.get("exact_sampling"))}

    for key, value in raw.items():
        if key == "phases":
            phases = dict(resolved["phases"])
            phases.update(value or {})
            resolved["phases"] = phases
        else:
            resolved[key] = value
    resolved["enabled"] = bool(resolved.get("enabled", False))
    resolved["max_qubits"] = int(resolved.get("max_qubits", 20))
    return resolved


def marginalize_probs_to_wires(probs: np.ndarray, n_qubits: int, wires: list[int] | None) -> np.ndarray:
    """Marginalize a full probability vector onto visible wires."""
    p = np.asarray(probs, dtype=np.float64)
    if wires is None:
        return p
    wires = [int(w) for w in wires]
    out = np.zeros(2 ** len(wires), dtype=np.float64)
    indices = np.arange(len(p), dtype=np.int64)
    bits = ((indices[:, None] >> np.arange(n_qubits, dtype=np.int64)) & 1).astype(np.int8)
    visible = bits[:, wires]
    visible_idx = np.sum(visible * (2 ** np.arange(len(wires), dtype=np.int64)), axis=1)
    np.add.at(out, visible_idx, p)
    return out


def reorder_probs_to_sample_indexing(probs: np.ndarray, n_qubits: int) -> np.ndarray:
    """Convert lexicographic circuit probabilities to little-endian sample indices."""
    p = np.asarray(probs, dtype=np.float64)
    indices = np.arange(len(p), dtype=np.int64)
    bits = ((indices[:, None] >> np.arange(n_qubits, dtype=np.int64)) & 1).astype(np.int8)
    reversed_indices = np.sum(bits[:, ::-1] * (2 ** np.arange(n_qubits, dtype=np.int64)), axis=1)
    out = np.empty_like(p)
    out[reversed_indices] = p
    return out


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
    sigma: float | list | dict
    shots: int
    rng_seed: int
    skip_sampling: bool = False
    final_eval_sampling: bool = False
    validity_fn: Callable | None = None
    coverage_fn: Callable | None = None
    exact_probs: np.ndarray | None = None
    generation_eval_fn: Callable | None = None
    exact_metrics: dict | None = None

    @property
    def sampling_enabled(self) -> bool:
        return not self.skip_sampling

    @property
    def final_sampling_enabled(self) -> bool:
        return (not self.skip_sampling) or self.final_eval_sampling

    def rng_for_step(self, step: int) -> np.random.Generator:
        return np.random.default_rng(self.rng_seed + step * 7919)

    def exact_metrics_enabled(self, phase: str) -> bool:
        cfg = resolve_exact_metrics_config(self.exact_metrics)
        return bool(cfg["enabled"] and cfg["phases"].get(phase, False))

    def evaluate_samples(self, samples: np.ndarray, model_probs: np.ndarray | None = None) -> dict:
        return evaluate_samples(
            self.x_train,
            samples,
            self.sigma,
            self.validity_fn,
            self.coverage_fn,
            exact_probs=self.exact_probs,
            generation_eval_fn=self.generation_eval_fn,
            model_probs=model_probs,
        )

    def evaluate_exact_probs(self, model_probs: np.ndarray) -> dict:
        return compute_exact_distribution_metrics(self.x_train, self.exact_probs, model_probs)

    def exact_model_probs(self, circuit, params, wires: list | None = None) -> np.ndarray | None:
        cfg = resolve_exact_metrics_config(self.exact_metrics)
        return _safe_model_probs(circuit, params, wires, cfg)

    def exact_ensemble_probs(self, ensemble, weights_override: np.ndarray | None = None) -> np.ndarray | None:
        cfg = resolve_exact_metrics_config(self.exact_metrics)
        if not _can_compute_exact_probs(ensemble.iqp_circuit, cfg):
            return None
        weights_input = ensemble.weights if weights_override is None else weights_override
        weights = np.asarray(weights_input, dtype=np.float64)
        if weights.size == 0 or len(weights) != len(ensemble.models):
            _warn_exact("ensemble weights do not match model count")
            return None
        total = float(weights.sum())
        if total <= 0:
            _warn_exact("ensemble weights sum to a non-positive value")
            return None
        weights = weights / total
        combined = None
        for weight, params in zip(weights, ensemble.models):
            probs = _safe_model_probs(ensemble.iqp_circuit, params, ensemble.wires, cfg, prechecked=True)
            if probs is None:
                return None
            combined = weight * probs if combined is None else combined + weight * probs
        if combined is None:
            return None
        return combined / combined.sum()

    def evaluate_ensemble_training_mmd(self, ensemble, ground_truth: np.ndarray | None = None) -> float:
        return compute_ensemble_training_mmd(
            ensemble,
            self.x_train if ground_truth is None else ground_truth,
        )

    def sample_and_evaluate_ensemble(self, ensemble, step: int) -> tuple[np.ndarray, dict]:
        samples = ensemble.sample(self.shots, self.rng_for_step(step))
        model_probs = self.exact_ensemble_probs(ensemble) if self.exact_metrics_enabled("steps") else None
        return samples, self.evaluate_samples(samples, model_probs=model_probs)

    def sample_and_evaluate_circuit(self, circuit, params, wires: list | None = None) -> tuple[np.ndarray, dict]:
        try:
            samples = circuit.sample(params, shots=self.shots, wires=wires)
        except TypeError:
            samples = circuit.sample(params, shots=self.shots)
        if wires is not None and samples.shape[1] != len(wires):
            samples = samples[:, wires]
        model_probs = self.exact_model_probs(circuit, params, wires) if self.exact_metrics_enabled("baseline") else None
        return samples, self.evaluate_samples(samples, model_probs=model_probs)

    def analytical_mmd_stats(self, value: float, model_probs: np.ndarray | None = None) -> dict:
        stats = {"mmd": value}
        if model_probs is not None:
            stats.update(self.evaluate_exact_probs(model_probs))
        return stats


def _can_compute_exact_probs(circuit, cfg: dict) -> bool:
    if not cfg.get("enabled", False):
        return False
    if not hasattr(circuit, "probs"):
        _warn_exact("circuit has no probs method")
        return False
    if bool(getattr(circuit, "bitflip", False)):
        _warn_exact("probs are unsupported for bitflip circuits")
        return False
    n_qubits = int(getattr(circuit, "n_qubits", 0))
    if n_qubits <= 0:
        _warn_exact("circuit n_qubits is unavailable")
        return False
    if n_qubits > int(cfg.get("max_qubits", 20)):
        _warn_exact(f"circuit has {n_qubits} qubits, above exact_metrics.max_qubits={cfg.get('max_qubits')}")
        return False
    return True


def _safe_model_probs(circuit, params, wires: list | None, cfg: dict, prechecked: bool = False) -> np.ndarray | None:
    if not prechecked and not _can_compute_exact_probs(circuit, cfg):
        return None
    try:
        probs = np.asarray(circuit.probs(params), dtype=np.float64)
    except Exception as exc:
        _warn_exact(f"failed to compute circuit probabilities: {exc}")
        return None
    if probs.ndim != 1 or probs.size == 0 or not np.all(np.isfinite(probs)):
        _warn_exact("circuit probabilities are empty, non-1D, or non-finite")
        return None
    total = float(probs.sum())
    if total <= 0:
        _warn_exact("circuit probabilities sum to a non-positive value")
        return None
    probs = probs / total
    n_qubits = int(getattr(circuit, "n_qubits", round(np.log2(len(probs)))))
    if len(probs) != 2 ** n_qubits:
        _warn_exact("circuit probability vector length is not 2**n_qubits")
        return None
    probs = reorder_probs_to_sample_indexing(probs, n_qubits)
    probs = marginalize_probs_to_wires(probs, n_qubits, wires)
    probs = probs / probs.sum()
    return probs


def _warn_exact(reason: str) -> None:
    warnings.warn(f"Skipping exact distribution metrics: {reason}", RuntimeWarning, stacklevel=2)
