"""Plot MMD progression and final metrics for the best HPO trial from saved data.

Usage:
    uv run python scripts/plot_best_hpo_model.py <hpo_output_dir>

Example:
    uv run python scripts/plot_best_hpo_model.py out/hpo/mnist_100q_1class_20260608_155253
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_BEST_TRIAL_DIRS = {
    0: "trial_0000",
    1: "trial_0001",
    2: "trial_0010",
    3: "trial_0011",
    4: "trial_0100",
    5: "trial_0101",
    6: "trial_0110",
    7: "trial_0111",
    8: "trial_1000",
    9: "trial_1001",
    10: "trial_1010",
    11: "trial_1011",
    12: "trial_1100",
    13: "trial_1101",
    14: "trial_1110",
}


def _trial_dir(num: int) -> str:
    return f"trial_{num:04d}"


def load_results_csv(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def plot_mmd_progression(rows: list[dict], summary: dict, output_dir: Path) -> None:
    boost_steps = []
    mmd_vals = []
    for r in rows:
        step = int(r.get("step", -99))
        if step < 0:
            continue
        mmd_str = r.get("mmd", "")
        if mmd_str and mmd_str.strip():
            boost_steps.append(step)
            mmd_vals.append(float(mmd_str))

    if not boost_steps:
        print("No boosting step data found in results.csv")
        return

    boost_steps = np.array(boost_steps)
    mmd_vals = np.array(mmd_vals)

    final_stats = summary["best_user_attrs"]["final_stats"]
    train_mmd = final_stats.get("mmd")
    test_mmd = final_stats.get("test_mmd")
    test_mmd_fcfw = final_stats.get("test_mmd_fcfw")

    best_value = summary["best_value"]
    best_trial = summary["best_trial"]
    best_params = summary["best_params"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # -- Left: MMD progression --
    ax1.plot(boost_steps, mmd_vals, "bo-", markersize=8, linewidth=2, label="Training MMD²")
    if train_mmd is not None:
        ax1.axhline(y=train_mmd, color="blue", linestyle="--", alpha=0.5, label=f"Final train: {train_mmd:.6f}")
    if test_mmd is not None:
        ax1.axhline(y=test_mmd, color="red", linestyle="--", alpha=0.7, label=f"Test MMD²: {test_mmd:.6f}")
    if test_mmd_fcfw is not None:
        ax1.axhline(y=test_mmd_fcfw, color="green", linestyle=":", alpha=0.7, label=f"Test (FCFW): {test_mmd_fcfw:.6f}")
    ax1.set_xlabel("Boosting step")
    ax1.set_ylabel("MMD²")
    ax1.set_title(f"MMD² Progression (Trial {best_trial})")
    ax1.legend(fontsize="small")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(boost_steps)

    # -- Right: Final metrics bar --
    metrics = {}
    labels = []
    values = []
    if train_mmd is not None:
        metrics["Train MMD²"] = train_mmd
        labels.append("Train MMD²")
        values.append(train_mmd)
    if test_mmd is not None:
        metrics["Test MMD²"] = test_mmd
        labels.append("Test MMD²")
        values.append(test_mmd)
    if test_mmd_fcfw is not None:
        metrics["Test (FCFW)"] = test_mmd_fcfw
        labels.append("Test (FCFW)")
        values.append(test_mmd_fcfw)

    colors_bar = ["#2ca02c", "#1f77b4", "#ff7f0e"][:len(values)]
    x_pos = np.arange(len(values))
    ax2.bar(x_pos, values, color=colors_bar, width=0.5, edgecolor="white")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("MMD²")
    ax2.set_title("Final Metrics")
    ax2.grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(values):
        ax2.text(i, v + max(values) * 0.02, f"{v:.6f}", ha="center", va="bottom", fontsize=9)

    # -- Title with best params --
    param_str = " | ".join(f"{k}={v}" for k, v in best_params.items())
    fig.suptitle(
        f"{summary['study_name']} | Trial {best_trial} | obj={best_value:.6f}\n{param_str}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    pdf_path = output_dir / "mmd_progression.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    png_path = output_dir / "mmd_progression.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {png_path}")
    plt.close(fig)


def print_summary(rows: list[dict], summary: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Study: {summary['study_name']}")
    print(f"Best trial: {summary['best_trial']}")
    print(f"Objective ({summary['objective_metric']}): {summary['best_value']:.6f}")
    print(f"\nBest params:")
    for k, v in summary["best_params"].items():
        print(f"  {k}: {v}")
    final_stats = summary["best_user_attrs"]["final_stats"]
    print(f"\nFinal metrics:")
    for k, v in final_stats.items():
        print(f"  {k}: {v:.6f}")
    print(f"\nPer-step MMD²:")
    for r in rows:
        step = r.get("step", "")
        mmd = r.get("mmd", "")
        test_mmd = r.get("test_mmd", "")
        label = r.get("model_label", "")
        parts = [f"  step={step:>3}"]
        if mmd:
            parts.append(f"mmd²={float(mmd):.6f}")
        if test_mmd:
            parts.append(f"test_mmd²={float(test_mmd):.6f}")
        if label:
            parts.append(f"({label})")
        print("  ".join(parts))
    print(f"{'='*60}\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/plot_best_hpo_model.py <hpo_output_dir>")
        sys.exit(1)

    hpo_dir = Path(sys.argv[1])
    if not hpo_dir.exists():
        print(f"HPO output directory not found: {hpo_dir}")
        sys.exit(1)

    summary_path = hpo_dir / "study_summary.json"
    if not summary_path.exists():
        print(f"Missing: {summary_path}")
        sys.exit(1)

    summary = json.loads(summary_path.read_text())
    best_trial_num = summary["best_trial"]
    trial_dir = hpo_dir / "trials" / _trial_dir(best_trial_num)
    results_csv = trial_dir / "results.csv"

    if not results_csv.exists():
        print(f"Missing: {results_csv}")
        sys.exit(1)

    rows = load_results_csv(results_csv)
    print_summary(rows, summary)

    output_dir = hpo_dir / "best_plot"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_mmd_progression(rows, summary, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
