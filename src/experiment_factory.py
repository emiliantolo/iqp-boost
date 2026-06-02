"""Dataset and plotting factories for config-driven experiments."""

from __future__ import annotations

from typing import Callable

from src.boltzmann_visualization import generate_boltzmann_visualizations
from src.datasets.hamming_balls import HammingBallsDataset
from src.datasets.hopfield import HopfieldDataset

SUPPORTED_DATASETS = ("hopfield", "hamming_balls")
RESERVED_DATASETS = ()


def _unsupported_dataset_error(dataset_key: str) -> ValueError:
    supported = ", ".join(SUPPORTED_DATASETS)
    message = f"Unsupported dataset '{dataset_key}'. Supported datasets: {supported}."
    if RESERVED_DATASETS:
        message += f" Reserved future datasets: {', '.join(RESERVED_DATASETS)}."
    return ValueError(message)


def _resolve_plot_kind(dataset_key: str, plot_spec: dict | None) -> str:
    if plot_spec and "kind" in plot_spec:
        return str(plot_spec["kind"]).lower()

    defaults = {
        "hamming_balls": "hamming_balls_mode_evolution",
        "hopfield": "boltzmann_summary",
    }
    return defaults.get(dataset_key, "none")


def _build_custom_viz(dataset_key: str, dataset_obj, plot_spec: dict | None) -> Callable | None:
    kind = _resolve_plot_kind(dataset_key, plot_spec)

    if kind == "none":
        return None
    if kind == "hamming_balls_mode_evolution":
        return None
    if kind == "boltzmann_summary":
        return lambda output, x_train, baseline_samples, final_samples, per_model_samples, weights: generate_boltzmann_visualizations(
            output, x_train, baseline_samples, final_samples, per_model_samples, weights, dataset_obj
        )

    raise ValueError(f"Unknown plot kind '{kind}'.")


def build_dataset_bundle(dataset_spec: dict, config: dict, plot_spec: dict | None = None) -> dict:
    """Create the Hopfield dataset, generated training set, metrics, and optional viz callback."""
    if not isinstance(dataset_spec, dict):
        raise ValueError("Each run requires a 'dataset' object with at least a 'name'.")

    dataset_key = str(dataset_spec.get("name", "")).strip().lower()
    params = dataset_spec.get("params", {})

    if dataset_key == "hamming_ball":
        raise ValueError("Unsupported dataset 'hamming_ball'. Use 'hamming_balls'.")
    if dataset_key not in SUPPORTED_DATASETS:
        raise _unsupported_dataset_error(dataset_key)

    train_samples = int(config.get("train_samples", 1000))
    data_seed = int(config.get("data_seed", 0))

    if dataset_key == "hamming_balls":
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
        )
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        return {
            "dataset_name": f"Hamming Balls (n={n_qubits}, K={K}, p={p})",
            "x_train": x_train,
            "validity_fn": None,
            "coverage_fn": None,
            "top_k_tvd_fn": getattr(ds, "top_k_tvd", None),
            "custom_viz_fn": _build_custom_viz(dataset_key, ds, plot_spec),
            "exact_probs": getattr(ds, "probs", None),
            "generation_eval_fn": getattr(ds, "evaluate_generation", None),
        }

    n_qubits = int(params.get("n_qubits", 16))
    n_patterns = int(params.get("n_patterns", 5))
    beta = float(params.get("beta", 2.0))
    pattern_seed = int(params.get("pattern_seed", 0))
    max_exact_states = int(params.get("max_exact_states", 2**20))
    mcmc_burn_in = int(params.get("mcmc_burn_in", 256))
    mcmc_thinning = int(params.get("mcmc_thinning", 16))
    mcmc_sweeps_per_sample = int(params.get("mcmc_sweeps_per_sample", 1))
    test_samples = int(params.get("test_samples", 0))
    train_split_ratio = float(params.get("train_split_ratio", 0.8)) if test_samples > 0 else None

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

    if train_split_ratio is not None:
        total_samples = train_samples + test_samples
        x_train = ds.generate(n_samples=total_samples, seed=data_seed, split="train")
        x_test = ds.generate(split="test")
    else:
        x_train = ds.generate(n_samples=train_samples, seed=data_seed)
        x_test = None

    bundle = {
        "dataset_name": f"Hopfield ({n_qubits}q, {n_patterns}p)",
        "x_train": x_train,
        "validity_fn": None,
        "coverage_fn": None,
        "top_k_tvd_fn": getattr(ds, "top_k_tvd", None),
        "custom_viz_fn": _build_custom_viz(dataset_key, ds, plot_spec),
        "exact_probs": getattr(ds, "probs", None),
        "generation_eval_fn": getattr(ds, "evaluate_generation", None),
    }
    if x_test is not None:
        bundle["x_test"] = x_test
    return bundle
