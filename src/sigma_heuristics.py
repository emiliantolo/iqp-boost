"""Sigma bandwidth heuristics for MMD kernel selection.

Three strategies for computing the Gaussian kernel bandwidth(s):

1. **median** -- Classic median heuristic (median of pairwise distances) scaled
   by user-provided factor(s).  Backward-compatible with existing configs.
2. **percentile** -- Use percentiles of the pairwise distance distribution as
   sigma values directly.  Data-adaptive, no manual ``sigma_factor`` needed.
3. **medoids** -- K-medoids clustering on the 1-D pairwise distance histogram.
   Finds the *natural distance scales* present in the data.

All methods subsample the training data to ``max_samples`` (default 1000) to
keep the O(m^2) pairwise computation bounded.
"""

from __future__ import annotations

import numpy as np
from iqpopt.gen_qml.utils import median_heuristic


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subsample_data(x: np.ndarray, max_samples: int = 1000,
                    seed: int = 42) -> np.ndarray:
    """Shuffle and cap the number of samples for pairwise computation."""
    rng = np.random.default_rng(seed)
    if len(x) <= max_samples:
        idx = rng.permutation(len(x))
        return x[idx]
    idx = rng.choice(len(x), size=max_samples, replace=False)
    return x[idx]


def _pairwise_hamming(x: np.ndarray) -> np.ndarray:
    """Upper-triangle pairwise Hamming distances as a flat 1-D array.

    For binary arrays this is equivalent to the L1 / Euclidean distance
    (since (x_i - x_j)^2 = |x_i - x_j| for bits).
    """
    x = np.asarray(x, dtype=np.float32)
    m = len(x)
    dists = []
    for i in range(m):
        diff = x[i + 1:] - x[i]
        dists.append(np.sqrt(np.sum(diff ** 2, axis=1)))
    return np.concatenate(dists) if dists else np.array([], dtype=np.float32)


def _kmedoids_1d(values: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """Simple 1-D k-medoids.  Returns sorted centroids.

    Uses the PAM-like swap algorithm on a 1-D sorted array.  No external
    dependency required.
    """
    values = np.sort(values)
    n = len(values)
    if n <= k:
        return np.unique(values)

    # Initialize centroids at evenly-spaced quantiles
    indices = np.linspace(0, n - 1, k, dtype=int)
    centroids = values[indices].copy()

    for _ in range(max_iter):
        # Assignment: each value to nearest centroid
        dists = np.abs(values[:, None] - centroids[None, :])  # (n, k)
        labels = np.argmin(dists, axis=1)

        # Update: medoid of each cluster = median element
        new_centroids = np.empty_like(centroids)
        for j in range(k):
            members = values[labels == j]
            if len(members) == 0:
                new_centroids[j] = centroids[j]
            else:
                new_centroids[j] = np.median(members)

        if np.allclose(new_centroids, centroids, atol=1e-10):
            break
        centroids = new_centroids

    return np.sort(centroids)


# ---------------------------------------------------------------------------
# Public API -- individual strategies
# ---------------------------------------------------------------------------

def compute_sigma_median(x_train: np.ndarray,
                         sigma_factor: list[float] | float = 0.5,
                         max_samples: int = 1000,
                         seed: int = 42) -> float | list[float]:
    """Median heuristic * factor(s).  Legacy behavior."""
    x_sub = _subsample_data(x_train, max_samples, seed)
    base = float(median_heuristic(x_sub))
    print(f"[sigma:median] sigma_base={base:.6f}  (on {len(x_sub)} samples)")

    if isinstance(sigma_factor, list):
        sigma = [f * base for f in sigma_factor]
    else:
        sigma = float(sigma_factor) * base
    return sigma


def compute_sigma_percentile(x_train: np.ndarray,
                             percentiles: list[float] | None = None,
                             max_samples: int = 1000,
                             seed: int = 42) -> list[float]:
    """Percentiles of pairwise Hamming distance distribution."""
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    x_sub = _subsample_data(x_train, max_samples, seed)
    dists = _pairwise_hamming(x_sub)

    if len(dists) == 0:
        raise ValueError("Not enough samples to compute pairwise distances")

    sigma = np.percentile(dists, percentiles).tolist()
    # Remove any zero or duplicate values
    sigma = sorted(set(s for s in sigma if s > 0))
    if not sigma:
        raise ValueError("All percentile sigma values are zero -- data may be constant")

    median_val = float(np.median(dists))
    print(f"[sigma:percentile] median_dist={median_val:.6f}, "
          f"{len(sigma)} sigma values from percentiles {percentiles}  "
          f"(on {len(x_sub)} samples)")
    return sigma


def compute_sigma_medoids(x_train: np.ndarray,
                          n_sigmas: int = 5,
                          max_samples: int = 1000,
                          seed: int = 42) -> list[float]:
    """K-medoids clustering on pairwise distance distribution."""
    x_sub = _subsample_data(x_train, max_samples, seed)
    dists = _pairwise_hamming(x_sub)

    if len(dists) == 0:
        raise ValueError("Not enough samples to compute pairwise distances")

    # Remove exact zeros (same-sample self-overlaps)
    dists = dists[dists > 0]
    if len(dists) == 0:
        raise ValueError("All pairwise distances are zero -- data may be constant")

    centroids = _kmedoids_1d(dists, k=n_sigmas)
    sigma = [float(c) for c in centroids if c > 0]

    if not sigma:
        raise ValueError("K-medoids produced only zero centroids")

    median_val = float(np.median(dists))
    print(f"[sigma:medoids] median_dist={median_val:.6f}, "
          f"{len(sigma)} sigma centroids via k-medoids  "
          f"(on {len(x_sub)} samples)")
    return sigma


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def compute_sigma(config: dict, x_train: np.ndarray,
                  seed: int = 42) -> float | list[float]:
    """Compute sigma bandwidth(s) from config.

    Resolution order:
    1. ``sigma_heuristic`` object/string  -> use specified method
    2. ``sigma_factor`` present           -> median * factors (backward compat)
    3. ``sigma`` present                  -> use literal value(s)
    4. fallback                           -> median * 0.4
    """
    heuristic = config.get('sigma_heuristic')

    # --- New-style: sigma_heuristic object or string ---
    if heuristic is not None:
        if isinstance(heuristic, str):
            heuristic = {'method': heuristic}

        method = heuristic.get('method', 'median')
        max_samples = int(heuristic.get('max_samples', 1000))

        if method == 'median':
            sf = heuristic.get('sigma_factor', config.get('sigma_factor', [0.5]))
            return compute_sigma_median(x_train, sigma_factor=sf,
                                        max_samples=max_samples, seed=seed)

        if method == 'percentile':
            pcts = heuristic.get('percentiles', [10, 25, 50, 75, 90])
            return compute_sigma_percentile(x_train, percentiles=pcts,
                                            max_samples=max_samples, seed=seed)

        if method == 'medoids':
            n_sigmas = int(heuristic.get('n_sigmas', 5))
            return compute_sigma_medoids(x_train, n_sigmas=n_sigmas,
                                         max_samples=max_samples, seed=seed)

        raise ValueError(f"Unknown sigma_heuristic method: {method}")

    # --- Legacy: sigma_factor ---
    if 'sigma_factor' in config:
        return compute_sigma_median(x_train, sigma_factor=config['sigma_factor'],
                                    seed=seed)

    # --- Direct sigma value ---
    if 'sigma' in config:
        return config['sigma']

    # --- Fallback ---
    return compute_sigma_median(x_train, sigma_factor=0.4, seed=seed)
