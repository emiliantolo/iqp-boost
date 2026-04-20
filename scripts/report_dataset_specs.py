#!/usr/bin/env python3
"""Report dataset specification metrics across configured sizes.

This script focuses on metrics commonly used in generative-model benchmarking:
- nominal support (exact when available, otherwise observed lower bound),
- support fraction in ambient space,
- Shannon entropy and perplexity (effective support),
- participation ratio (inverse Simpson index),
- top-1 mass concentration.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.bas import BarsAndStripesDataset
from src.datasets.blobs import BlobsDataset
from src.datasets.dwave import DWaveDataset
from src.datasets.genomic import GenomicDataset
from src.datasets.ising import FrustratedIsingDataset
from src.datasets.mnist import BinarizedMNISTDataset
from src.datasets.pennylane_bas import PennylaneBASDataset
from src.datasets.pennylane_hm import PennylaneHMDataset
from src.datasets.pennylane_ising import PennylaneIsingDataset
from src.datasets.scale_free import ScaleFreeDataset


@dataclass
class DatasetSpec:
    dataset: str
    dims: str
    n_qubits: int
    beta: float | None
    ambient_states: int
    support_kind: str
    support_size: int
    support_fraction: float
    n_reference_samples: int
    n_unique_observed: int
    entropy_bits: float
    perplexity: float
    participation_ratio: float
    top1_mass: float
    top10_mass: float


def _rows_to_tuples(arr: np.ndarray) -> list[tuple[int, ...]]:
    arr = np.asarray(arr, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D sample array, got shape {arr.shape}")
    return [tuple(row.tolist()) for row in arr]


def _distribution_metrics_from_samples(samples: np.ndarray) -> tuple[int, float, float, float, float]:
    rows = _rows_to_tuples(samples)
    if not rows:
        return 0, 0.0, 0.0, 0.0, 0.0

    counts = Counter(rows)
    total = float(len(rows))
    probs = np.asarray([c / total for c in counts.values()], dtype=np.float64)
    entropy_bits = float(-np.sum(probs * np.log2(probs + 1e-300)))
    perplexity = float(2.0 ** entropy_bits)
    participation_ratio = float(1.0 / np.sum(probs ** 2))
    top1_mass = float(np.max(probs))
    return len(counts), entropy_bits, perplexity, participation_ratio, top1_mass


def _distribution_metrics_from_probs(probs: np.ndarray) -> tuple[int, float, float, float, float]:
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    p = p / p.sum()
    support_size = int(p.size)
    entropy_bits = float(-np.sum(p * np.log2(p + 1e-300)))
    perplexity = float(2.0 ** entropy_bits)
    participation_ratio = float(1.0 / np.sum(p ** 2))
    top1_mass = float(np.max(p))
    return support_size, entropy_bits, perplexity, participation_ratio, top1_mass


def _top_k_mass_from_samples(samples: np.ndarray, k: int = 10) -> float:
    rows = _rows_to_tuples(samples)
    if not rows:
        return 0.0
    counts = Counter(rows)
    probs = np.asarray(sorted(counts.values(), reverse=True), dtype=np.float64)
    probs = probs / probs.sum()
    return float(np.sum(probs[:k]))


def _top_k_mass_from_probs(probs: np.ndarray, k: int = 10) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    p = np.sort(p)[::-1]
    return float(np.sum(p[:k]))


def _support_from_valid_patterns(dataset) -> int | None:
    ensure = getattr(dataset, "_ensure_valid_patterns", None)
    if callable(ensure):
        ensure()
    support = getattr(dataset, "_valid_patterns", None)
    if support is None:
        return None
    return int(len(support))


def _instantiate(dataset: str, rows: int, cols: int, args: argparse.Namespace):
    if dataset == "bas":
        return BarsAndStripesDataset(height=rows, width=cols)
    if dataset == "blobs":
        return BlobsDataset()
    if dataset == "dwave":
        return DWaveDataset(rows=rows, cols=cols)
    if dataset == "genomic":
        return GenomicDataset(rows=rows, cols=cols)
    if dataset == "ising":
        return FrustratedIsingDataset(rows=rows, cols=cols, beta=args.beta, j_seed=args.j_seed, batch_size=args.ising_batch_size)
    if dataset == "mnist":
        return BinarizedMNISTDataset(rows=rows, cols=cols, digit=args.digit, reduction=args.reduction)
    if dataset == "pennylane_bas":
        return PennylaneBASDataset(rows=rows, cols=cols)
    if dataset == "pennylane_hm":
        return PennylaneHMDataset(rows=rows, cols=cols)
    if dataset == "pennylane_ising":
        return PennylaneIsingDataset(rows=rows, cols=cols)
    if dataset == "scale_free":
        return ScaleFreeDataset(rows=rows, cols=cols)
    raise ValueError(f"Unsupported dataset in grid reports: {dataset}")


def _extract_grid_targets(config_dir: Path) -> list[tuple[str, int, int]]:
    pattern = re.compile(r"(.+)_([0-9]+)x([0-9]+)_grid\.json$")
    targets: set[tuple[str, int, int]] = set()
    for path in config_dir.glob("*_grid.json"):
        m = pattern.match(path.name)
        if not m:
            continue
        dataset, rs, cs = m.group(1), int(m.group(2)), int(m.group(3))
        targets.add((dataset, rs, cs))

    filtered = []
    for dataset, rows, cols in sorted(targets):
        # Requested by user: keep only 16-qubit PennyLane BAS.
        if dataset == "pennylane_bas" and rows * cols != 16:
            continue
        filtered.append((dataset, rows, cols))
    return filtered


def _analyze(dataset: str, rows: int, cols: int, args: argparse.Namespace) -> DatasetSpec:
    ds = _instantiate(dataset, rows, cols, args)
    n_qubits = int(getattr(ds, "n_qubits", rows * cols))
    ambient = int(2 ** n_qubits)

    if dataset == "ising":
        support_size, entropy_bits, perplexity, participation_ratio, top1_mass = _distribution_metrics_from_probs(ds.probs)
        top10_mass = _top_k_mass_from_probs(ds.probs, k=10)
        support_kind = "exact-full-support"
        n_reference_samples = int(len(ds.probs))
        n_unique_observed = support_size
    else:
        support_exact = _support_from_valid_patterns(ds)
        if support_exact is not None:
            support_kind = "exact-valid-patterns"
            support_size = int(support_exact)
        else:
            loaded = getattr(ds, "data", None)
            if loaded is not None:
                support_kind = "exact-loaded-unique"
                support_size = int(np.unique(np.asarray(loaded, dtype=np.int8), axis=0).shape[0])
            else:
                support_kind = f"observed-lower-bound-{args.samples}"
                observed = ds.generate(n_samples=args.samples, seed=args.seed)
                support_size = int(np.unique(np.asarray(observed, dtype=np.int8), axis=0).shape[0])

        ref = getattr(ds, "data", None)
        if ref is None:
            ref = ds.generate(n_samples=args.samples, seed=args.seed)

        n_unique_observed, entropy_bits, perplexity, participation_ratio, top1_mass = _distribution_metrics_from_samples(ref)
        top10_mass = _top_k_mass_from_samples(ref, k=10)
        n_reference_samples = int(len(ref))

    return DatasetSpec(
        dataset=dataset,
        dims=f"{rows}x{cols}",
        n_qubits=n_qubits,
        beta=float(args.beta) if dataset == "ising" else None,
        ambient_states=ambient,
        support_kind=support_kind,
        support_size=int(support_size),
        support_fraction=float(support_size / ambient if ambient > 0 else 0.0),
        n_reference_samples=n_reference_samples,
        n_unique_observed=n_unique_observed,
        entropy_bits=entropy_bits,
        perplexity=perplexity,
        participation_ratio=participation_ratio,
        top1_mass=top1_mass,
        top10_mass=top10_mass,
    )


def _print_table(specs: list[DatasetSpec]) -> None:
    header = [
        "dataset",
        "dims",
        "n",
        "beta",
        "support",
        "supp_frac",
        "H(bits)",
        "perplex",
        "PR",
        "top1",
        "top10",
        "kind",
    ]
    rows = []
    for s in specs:
        rows.append(
            [
                s.dataset,
                s.dims,
                str(s.n_qubits),
                "-" if s.beta is None else f"{s.beta:g}",
                f"{s.support_size:,}",
                f"{s.support_fraction:.4g}",
                f"{s.entropy_bits:.3f}",
                f"{s.perplexity:.2f}",
                f"{s.participation_ratio:.2f}",
                f"{s.top1_mass:.4f}",
                f"{s.top10_mass:.4f}",
                s.support_kind,
            ]
        )

    widths = [max(len(str(cell)) for cell in col) for col in zip(header, *rows)]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


def _to_latex(specs: list[DatasetSpec], caption: str) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllrrrrrrrl}",
        "\\hline",
        "Dataset & Dims & $n$ & $\\beta$ & $|\\mathcal{S}|$ & $|\\mathcal{S}|/2^n$ & $H$ (bits) & $2^H$ & PR & Top-1 & Top-10 \\\\",
        "\\hline",
    ]
    for s in specs:
        lines.append(
            f"{s.dataset} & {s.dims} & {s.n_qubits} & "
            f"{'-' if s.beta is None else f'{s.beta:g}'} & {s.support_size} & "
            f"{s.support_fraction:.4g} & {s.entropy_bits:.3f} & {s.perplexity:.2f} & "
            f"{s.participation_ratio:.2f} & {s.top1_mass:.4f} & {s.top10_mass:.4f} \\\\" 
        )
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute dataset spec metrics across configured grid sizes.")
    parser.add_argument("--config-dir", default="configs/grids", help="Directory containing *_grid.json files.")
    parser.add_argument("--samples", type=int, default=20000, help="Reference sample count when a dataset is not pre-loaded.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.5, help="Ising inverse temperature.")
    parser.add_argument("--j-seed", type=int, default=0, help="Ising coupling seed.")
    parser.add_argument("--ising-batch-size", type=int, default=32768, help="Batch size for exact Ising Boltzmann computation.")
    parser.add_argument("--digit", type=int, default=None, help="Optional MNIST digit filter.")
    parser.add_argument("--reduction", choices=("spatial", "pca"), default="spatial")
    parser.add_argument("--json", dest="json_path", default=None, help="Optional output JSON path.")
    parser.add_argument("--latex", dest="latex_path", default=None, help="Optional output LaTeX table path.")
    parser.add_argument(
        "--caption",
        default=(
            "Dataset support and concentration metrics across configured sizes. "
            "PR denotes participation ratio (inverse Simpson index)."
        ),
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    targets = _extract_grid_targets(config_dir)
    specs = [_analyze(dataset, rows, cols, args) for dataset, rows, cols in targets]

    _print_table(specs)

    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(s) for s in specs], indent=2) + "\n")
        print(f"\nSaved JSON: {out}")

    if args.latex_path:
        out = Path(args.latex_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_to_latex(specs, caption=args.caption))
        print(f"Saved LaTeX: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())