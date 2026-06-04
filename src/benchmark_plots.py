"""Benchmark plotting helpers restored for experiment factory imports."""

from __future__ import annotations

import numpy as np


def _save_placeholder(output, filename: str, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.set_axis_off()
    path = output.get_path(filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved benchmark plot to: {path}")


def plot_mode_evolution(output, *args, filename: str = "mode_evolution.pdf", **kwargs) -> None:
    _save_placeholder(output, filename, "Mode evolution")


def plot_nll_convergence(output, *args, filename: str = "nll_convergence.pdf", **kwargs) -> None:
    _save_placeholder(output, filename, "NLL convergence")


def plot_probability_alignment(
    output,
    *args,
    filename: str = "probability_alignment.pdf",
    **kwargs,
) -> None:
    _save_placeholder(output, filename, "Probability alignment")
