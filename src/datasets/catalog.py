"""Catalog of supported config-driven datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.datasets.boltzmann_visualization import generate_boltzmann_visualizations
from src.datasets.hamming_balls import HammingBallsDataset
from src.datasets.hopfield import HopfieldDataset
from src.core import EvaluationPolicy


@dataclass(frozen=True)
class DatasetBundle:
    """Catalog-owned dataset integration consumed by experiment runners."""

    dataset_name: str
    x_train: np.ndarray
    validity_fn: Callable | None = None
    coverage_fn: Callable | None = None
    top_k_tvd_fn: Callable | None = None
    custom_viz_fn: Callable | None = None
    exact_probs: np.ndarray | None = None
    generation_eval_fn: Callable | None = None
    x_test: np.ndarray | None = None

    @property
    def n_qubits(self) -> int:
        return int(self.x_train.shape[1])

    def build_evaluation_policy(
        self,
        sigma: float | list,
        shots: int,
        rng_seed: int,
        skip_sampling: bool = False,
        final_eval_sampling: bool = False,
    ) -> EvaluationPolicy:
        return EvaluationPolicy(
            x_train=self.x_train,
            sigma=sigma,
            shots=shots,
            rng_seed=rng_seed,
            skip_sampling=skip_sampling,
            final_eval_sampling=final_eval_sampling,
            validity_fn=self.validity_fn,
            coverage_fn=self.coverage_fn,
            exact_probs=self.exact_probs,
            generation_eval_fn=self.generation_eval_fn,
        )

    def run_custom_visualization(
        self,
        output,
        baseline_samples,
        final_samples,
        per_model_samples,
        weights,
    ) -> None:
        if self.custom_viz_fn is None:
            return
        self.custom_viz_fn(output, self.x_train, baseline_samples, final_samples, per_model_samples, weights)


@dataclass(frozen=True)
class DatasetCatalogEntry:
    key: str
    default_plot_kind: str
    builder: Callable[[dict[str, Any], dict[str, Any], dict[str, Any] | None], DatasetBundle]


def _unsupported_dataset_error(dataset_key: str) -> ValueError:
    supported = ", ".join(SUPPORTED_DATASETS)
    return ValueError(f"Unsupported dataset '{dataset_key}'. Supported datasets: {supported}.")


def _resolve_plot_kind(entry: DatasetCatalogEntry, plot_spec: dict[str, Any] | None) -> str:
    if plot_spec and "kind" in plot_spec:
        return str(plot_spec["kind"]).lower()
    return entry.default_plot_kind


def _build_hopfield_viz(dataset_obj: HopfieldDataset, plot_spec: dict[str, Any] | None) -> Callable | None:
    kind = _resolve_plot_kind(_CATALOG["hopfield"], plot_spec)
    if kind == "none":
        return None
    if kind == "boltzmann_summary":
        return lambda output, x_train, baseline_samples, final_samples, per_model_samples, weights: generate_boltzmann_visualizations(
            output, x_train, baseline_samples, final_samples, per_model_samples, weights, dataset_obj
        )
    raise ValueError(f"Unknown plot kind '{kind}'.")


def _build_hamming_balls_viz(plot_spec: dict[str, Any] | None) -> Callable | None:
    kind = _resolve_plot_kind(_CATALOG["hamming_balls"], plot_spec)
    if kind in {"none", "hamming_balls_mode_evolution"}:
        return None
    raise ValueError(f"Unknown plot kind '{kind}'.")


def _resolve_split(params: dict[str, Any], train_samples: int) -> tuple[int, float | None]:
    test_samples = int(params.get("test_samples", 0))
    if test_samples <= 0:
        return train_samples, None

    total_samples = train_samples + test_samples
    train_split_ratio = float(params.get("train_split_ratio", train_samples / total_samples))
    return total_samples, train_split_ratio


def _generate_bundle_samples(dataset_obj, total_samples: int, data_seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    if dataset_obj.train_split_ratio is None:
        return dataset_obj.generate(n_samples=total_samples, seed=data_seed), None
    x_train = dataset_obj.generate(n_samples=total_samples, seed=data_seed, split="train")
    x_test = dataset_obj.generate(split="test")
    return x_train, x_test


def _build_hamming_balls_bundle(
    params: dict[str, Any],
    config: dict[str, Any],
    plot_spec: dict[str, Any] | None,
) -> DatasetBundle:
    train_samples = int(config.get("train_samples", 1000))
    data_seed = int(config.get("data_seed", 0))
    total_samples, train_split_ratio = _resolve_split(params, train_samples)
    n_qubits = int(params.get("n_qubits", 16))
    K = int(params.get("K", 8))
    p = float(params.get("p", 0.1))
    pattern_seed = int(params.get("pattern_seed", 0))
    max_exact_states = int(params.get("max_exact_states", 2**20))

    ds = HammingBallsDataset(
        n_qubits=n_qubits,
        K=K,
        p=p,
        pattern_seed=pattern_seed,
        max_exact_states=max_exact_states,
        train_split_ratio=train_split_ratio,
    )
    x_train, x_test = _generate_bundle_samples(ds, total_samples, data_seed)

    return DatasetBundle(
        dataset_name=f"Hamming Balls (n={n_qubits}, K={K}, p={p})",
        x_train=x_train,
        top_k_tvd_fn=getattr(ds, "top_k_tvd", None),
        custom_viz_fn=_build_hamming_balls_viz(plot_spec),
        exact_probs=getattr(ds, "probs", None),
        generation_eval_fn=getattr(ds, "evaluate_generation", None),
        x_test=x_test,
    )


def _build_hopfield_bundle(
    params: dict[str, Any],
    config: dict[str, Any],
    plot_spec: dict[str, Any] | None,
) -> DatasetBundle:
    train_samples = int(config.get("train_samples", 1000))
    data_seed = int(config.get("data_seed", 0))
    total_samples, train_split_ratio = _resolve_split(params, train_samples)
    n_qubits = int(params.get("n_qubits", 16))
    n_patterns = int(params.get("n_patterns", 5))
    beta = float(params.get("beta", 2.0))
    pattern_seed = int(params.get("pattern_seed", 0))
    max_exact_states = int(params.get("max_exact_states", 2**20))
    mcmc_burn_in = int(params.get("mcmc_burn_in", 256))
    mcmc_thinning = int(params.get("mcmc_thinning", 16))
    mcmc_sweeps_per_sample = int(params.get("mcmc_sweeps_per_sample", 1))

    ds = HopfieldDataset(
        n_qubits=n_qubits,
        n_patterns=n_patterns,
        beta=beta,
        pattern_seed=pattern_seed,
        max_exact_states=max_exact_states,
        mcmc_burn_in=mcmc_burn_in,
        mcmc_thinning=mcmc_thinning,
        mcmc_sweeps_per_sample=mcmc_sweeps_per_sample,
        train_split_ratio=train_split_ratio,
    )

    x_train, x_test = _generate_bundle_samples(ds, total_samples, data_seed)

    return DatasetBundle(
        dataset_name=f"Hopfield ({n_qubits}q, {n_patterns}p)",
        x_train=x_train,
        top_k_tvd_fn=getattr(ds, "top_k_tvd", None),
        custom_viz_fn=_build_hopfield_viz(ds, plot_spec),
        exact_probs=getattr(ds, "probs", None),
        generation_eval_fn=getattr(ds, "evaluate_generation", None),
        x_test=x_test,
    )


_CATALOG = {
    "hopfield": DatasetCatalogEntry(
        key="hopfield",
        default_plot_kind="boltzmann_summary",
        builder=_build_hopfield_bundle,
    ),
    "hamming_balls": DatasetCatalogEntry(
        key="hamming_balls",
        default_plot_kind="hamming_balls_mode_evolution",
        builder=_build_hamming_balls_bundle,
    ),
}

SUPPORTED_DATASETS = tuple(_CATALOG)


def build_dataset_bundle(
    dataset_spec: dict[str, Any],
    config: dict[str, Any],
    plot_spec: dict[str, Any] | None = None,
) -> DatasetBundle:
    """Create dataset artifacts for a config-driven experiment."""
    if not isinstance(dataset_spec, dict):
        raise ValueError("Each run requires a 'dataset' object with at least a 'name'.")

    dataset_key = str(dataset_spec.get("name", "")).strip().lower()
    params = dataset_spec.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("Dataset 'params' must be an object when provided.")

    if dataset_key == "hamming_ball":
        raise ValueError("Unsupported dataset 'hamming_ball'. Use 'hamming_balls'.")

    entry = _CATALOG.get(dataset_key)
    if entry is None:
        raise _unsupported_dataset_error(dataset_key)

    return entry.builder(params, config, plot_spec)
