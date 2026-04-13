#!/usr/bin/env python3
"""Print summary tables for grid configs or completed suite results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _fmt_table(headers: list[str], rows: list[list[object]]) -> str:
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in str_rows)
    return "\n".join(lines)


def summarize_config(config_path: Path) -> str:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])

    combo_counter: Counter[tuple[str, int, int, str]] = Counter()

    for run in runs:
        cfg = run.get("config", {})
        circ = cfg.get("circuit_config", {})
        topology = str(circ.get("topology", ""))
        distance = int(circ.get("distance", -1))
        n_ancilla = int(circ.get("n_ancilla", 0))
        mode = str(circ.get("ancilla_topology_mode", ""))
        combo_counter[(topology, distance, n_ancilla, mode)] += 1

    summary_rows = [
        [top, dist, anc, mode, count]
        for (top, dist, anc, mode), count in sorted(
            combo_counter.items(),
            key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]),
        )
    ]

    lines = [
        f"Config: {config_path}",
        f"Total runs: {len(runs)}",
        f"Unique circuit combos: {len(summary_rows)}",
        "",
        _fmt_table(
            ["topology", "distance", "n_ancilla", "ancilla_topology_mode", "count"],
            summary_rows,
        ),
    ]

    return "\n".join(lines)


def _parse_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_run_circuit_info(run_dir: Path) -> tuple[str, int, int, str]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return "", -1, -1, ""

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    circ = payload.get("circuit_config", {})
    topology = str(circ.get("topology", ""))
    distance = int(circ.get("distance", -1))
    n_ancilla = int(circ.get("n_ancilla", -1))
    mode = str(circ.get("ancilla_topology_mode", ""))
    return topology, distance, n_ancilla, mode


def summarize_results(results_dir: Path) -> str:
    run_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()])
    rows = []
    skipped = []

    for run_dir in run_dirs:
        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            skipped.append((run_dir.name, "missing results.csv"))
            continue

        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            data = list(csv.DictReader(handle))

        if not data:
            skipped.append((run_dir.name, "empty results.csv"))
            continue

        baseline = next((r for r in data if str(r.get("step", "")).strip() == "-1"), None)
        ensemble_candidates = [r for r in data if _parse_float(r.get("step", "nan")) >= 0]

        if baseline is None or not ensemble_candidates:
            skipped.append((run_dir.name, "missing baseline or ensemble rows"))
            continue

        ensemble = max(ensemble_candidates, key=lambda r: int(float(r["step"])))
        topology, distance, n_ancilla, mode = _load_run_circuit_info(run_dir)

        baseline_mmd = _parse_float(baseline.get("mmd"))
        ensemble_mmd = _parse_float(ensemble.get("mmd"))
        delta_mmd = baseline_mmd - ensemble_mmd

        baseline_f1 = _parse_float(baseline.get("f_score"))
        ensemble_f1 = _parse_float(ensemble.get("f_score"))
        baseline_validity = _parse_float(baseline.get("validity"))
        ensemble_validity = _parse_float(ensemble.get("validity"))
        baseline_tvd = _parse_float(baseline.get("tvd"))
        ensemble_tvd = _parse_float(ensemble.get("tvd"))

        rows.append([
            run_dir.name,
            topology,
            distance,
            n_ancilla,
            mode,
            f"{baseline_mmd:.6f}",
            f"{ensemble_mmd:.6f}",
            f"{delta_mmd:+.6f}",
            f"{baseline_validity:.3f}",
            f"{ensemble_validity:.3f}",
            f"{ensemble_validity - baseline_validity:+.3f}",
            f"{baseline_tvd:.3f}",
            f"{ensemble_tvd:.3f}",
            f"{baseline_tvd - ensemble_tvd:+.3f}",
            f"{baseline_f1:.3f}",
            f"{ensemble_f1:.3f}",
        ])

    rows.sort(key=lambda r: r[0])

    improvements = 0
    regressions = 0
    ties = 0
    deltas = []
    for r in rows:
        d = float(r[7])
        deltas.append(d)
        if d > 0:
            improvements += 1
        elif d < 0:
            regressions += 1
        else:
            ties += 1

    header = [
        "run",
        "topo",
        "dist",
        "anc",
        "mode",
        "b_mmd",
        "e_mmd",
        "d_mmd",
        "b_val",
        "e_val",
        "d_val",
        "b_tvd",
        "e_tvd",
        "d_tvd",
        "b_f1",
        "e_f1",
    ]

    lines = [
        f"Results dir: {results_dir}",
        f"Completed runs: {len(rows)}",
        f"Skipped runs: {len(skipped)}",
        f"MMD improvements: {improvements} | regressions: {regressions} | ties: {ties}",
    ]

    if deltas:
        avg_delta = sum(deltas) / len(deltas)
        lines.append(f"Mean delta_mmd (baseline - ensemble): {avg_delta:+.6f}")

    lines.append("")
    lines.append(_fmt_table(header, rows if rows else [["(none)"] + [""] * (len(header) - 1)]))

    if skipped:
        lines.append("")
        lines.append("Skipped:")
        lines.append(_fmt_table(["run", "reason"], [[n, reason] for n, reason in skipped]))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a summary table for a grid config or a suite output directory."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="bas_3x3_grid.json",
        help="Path to config JSON or suite output directory",
    )
    args = parser.parse_args()

    target = Path(args.path).resolve()
    if not target.exists():
        raise SystemExit(f"Path not found: {target}")

    if target.is_dir():
        print(summarize_results(target))
    else:
        print(summarize_config(target))


if __name__ == "__main__":
    main()
