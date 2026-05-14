"""Sigma bandwidth heuristics for MMD kernel selection.

Seven strategies for computing the Gaussian kernel bandwidth(s):

1. **median** -- Classic median heuristic (median of pairwise distances) scaled
   by user-provided factor(s).  Backward-compatible with existing configs.
2. **percentile** -- Use percentiles of the pairwise distance distribution as
   sigma values directly.  Data-adaptive, no manual ``sigma_factor`` needed.
3. **medoids** -- K-medoids clustering on the 1-D pairwise distance histogram.
   Finds the *natural distance scales* present in the data.
4. **fourier** -- Deterministic, non-data-driven sigmas targeting specific
   k-body Fourier spectrum features of the Boolean hypercube.  No pairwise
   computations needed at all.
5. **optimized** -- Maximize MMD test power (SNR = MMD^2 / sqrt(Var)) between
   data and uniform reference via JAX gradient ascent on log(sigma).
   Initialized from fourier targets with diversity regularization.
   Based on Sutherland et al. (2017).
6. **k_order** -- Data-adaptive sigma selection by profiling the empirical
   k-order correlation structure. Ranks k-body orders by correlation
   strength (raw) or enrichment over IID null, then maps the top orders
   to sigma values via the Fourier inversion formula.
7. **fit_k_order** -- TVD-optimized sigma mixture matching. Minimizes the
   total variation distance between the data's empirical k-order profile
   and the theoretical bodyness distribution of the kernel mixture.

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


def compute_sigma_fourier(n: int, s: int) -> list[float]:
    """
    Computes 's' sigma bandwidths targeted at specific k-body Fourier depths,
    specifically calibrated for a Binomial Pauli sampling MMD estimator.
    """
    if s <= 0:
        return []

    # We mathematically cannot target exactly n/2 because it requires sigma=0.
    max_k = (n / 2) * (1 - 1e-5)

    if s == 1:
        targets = [max_k]
    else:
        # Geometrically space our k-body targets from k=1 up to max_k
        geom_points = np.geomspace(1, max_k, num=s)
        targets = [float(x) for x in geom_points]
        
    print(f"[sigma:fourier] Targeting expected k-body depths: {[round(t, 2) for t in targets]}")

    sigmas = []
    for k in targets:
        inner_term = 1 - (2 * k / n)
        sigma = np.sqrt(-1.0 / (2.0 * np.log(inner_term)))
        sigmas.append(float(sigma))

    return sorted(sigmas, reverse=True)


# ---------------------------------------------------------------------------
# k-order correlation helpers (ported from scratch/src)
# ---------------------------------------------------------------------------

def _sigma_from_expected_bodyness(
    n_qubits: int,
    expected_bodyness: float,
    sigma_floor: float = 1e-3,
) -> float:
    """Convert expected k-body order to sigma via the Fourier inversion formula.

    Maps the expected bodyness ``k`` to the Gaussian kernel bandwidth sigma
    that makes a Binomial(n, p) distribution with p = k/n concentrate at
    order ``k``.  Clamps to ``sigma_floor`` at the boundary.
    """
    if expected_bodyness <= 0:
        return float("inf")
    if expected_bodyness >= n_qubits / 2:
        return sigma_floor
    p = expected_bodyness / n_qubits
    return max(float(np.sqrt(-1.0 / (2.0 * np.log(1.0 - 2.0 * p)))), sigma_floor)


def _k_order_correlation_distribution(
    x: np.ndarray,
    max_k: int | None = None,
) -> np.ndarray:
    """Empirical active k-order correlations for k=1..max_k via weight histogramming.

    For each sample, counts the Hamming weight w (number of 1-bits), then
    computes the fraction of active k-subsets C(w,k)/C(n,k) averaged over
    samples.  This gives a fingerprint of the data's correlation structure.

    Uses JAX for efficient vectorised computation.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if x.shape[0] == 0:
        raise ValueError("x must contain at least one row")

    n_features = x.shape[1]
    if max_k is None:
        max_k = n_features
    upper_k = min(max_k, n_features)
    if upper_k == 0:
        return np.array([])

    X = jnp.asarray(x)
    float_dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    weights = jnp.sum(X, axis=1).astype(jnp.int32)
    hist = jnp.bincount(weights, length=n_features + 1).astype(float_dtype)
    normalized_hist = hist / jnp.asarray(X.shape[0], dtype=float_dtype)

    k_range = jnp.arange(1, upper_k + 1, dtype=float_dtype)[:, None]
    w_range = jnp.arange(n_features + 1, dtype=float_dtype)[None, :]
    valid = w_range >= k_range
    safe_w_range = jnp.where(valid, w_range, k_range)

    log_num = (gammaln(safe_w_range + 1.0)
               - gammaln(k_range + 1.0)
               - gammaln(safe_w_range - k_range + 1.0))
    log_den = (gammaln(jnp.asarray(n_features + 1, dtype=float_dtype))
               - gammaln(k_range + 1.0)
               - gammaln(jnp.asarray(n_features, dtype=float_dtype) - k_range + 1.0))
    ratios = jnp.where(valid, jnp.exp(log_num - log_den),
                        jnp.asarray(0.0, dtype=float_dtype))
    return np.asarray(ratios @ normalized_hist)


def _k_order_enrichment(
    x: np.ndarray,
    max_k: int | None = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """k-order correlations divided by the IID Bernoulli null p^k.

    Highlights orders that are *over-represented* relative to random,
    filtering out correlations that arise simply from the marginal bit rate.
    """
    import jax.numpy as jnp

    observed = _k_order_correlation_distribution(x, max_k)
    n_features = x.shape[1]
    if max_k is None:
        max_k = n_features
    upper_k = min(max_k, n_features)
    p = float(np.mean(x))
    null = np.array([p**k for k in range(1, upper_k + 1)])
    return observed / np.maximum(null, epsilon)


# ---------------------------------------------------------------------------
# 6. k_order -- data-adaptive sigma from correlation profile
# ---------------------------------------------------------------------------

def compute_sigma_k_order(
    x_train: np.ndarray,
    n_sigmas: int = 5,
    mode: str = "enriched",
    max_k: int | None = None,
    sigma_floor: float = 1e-3,
    max_samples: int = 1000,
    seed: int = 42,
) -> list[float]:
    """Pick sigmas targeting the k-body orders where the data has the most structure.

    Two modes are available:
      - ``"raw"``:  rank orders by raw correlation strength.
      - ``"enriched"``:  rank orders by enrichment over the IID null (default).
        This filters out trivial correlations from marginal bit rates.

    Args:
        x_train: Training data, shape (m, n_qubits).
        n_sigmas: Number of sigma bandwidths to produce.
        mode: ``"raw"`` or ``"enriched"`` ranking.
        max_k: Maximum k-body order to consider (default: n_qubits // 2).
        sigma_floor: Minimum sigma value (prevents degenerate sigmas).
        max_samples: Subsample x_train for efficiency.
        seed: Random seed.

    Returns:
        List of sigma values sorted descending.
    """
    x_sub = _subsample_data(x_train, max_samples=max_samples, seed=seed)
    n_qubits = x_sub.shape[1]

    if max_k is None:
        max_k = n_qubits // 2
    max_k = min(max_k, n_qubits // 2)

    if mode == "enriched":
        profile = _k_order_enrichment(x_sub, max_k=max_k)
    else:
        profile = _k_order_correlation_distribution(x_sub, max_k=max_k)

    # Rank orders 1..max_k by profile strength (descending), break ties by smaller k
    ranked_orders = sorted(
        range(1, len(profile) + 1),
        key=lambda k: (-float(profile[k - 1]), k),
    )
    selected_orders = ranked_orders[:n_sigmas]

    sigmas = [
        _sigma_from_expected_bodyness(n_qubits, float(k), sigma_floor)
        for k in selected_orders
    ]

    print(f"[sigma:k_order] mode={mode}, selected k-body orders: {selected_orders}")
    print(f"[sigma:k_order] sigmas: {[round(s, 6) for s in sorted(sigmas, reverse=True)]}")

    return sorted(sigmas, reverse=True)


# ---------------------------------------------------------------------------
# 7. fit_k_order -- TVD-optimized bodyness matching
# ---------------------------------------------------------------------------

def compute_sigma_fit_k_order(
    x_train: np.ndarray,
    n_sigmas: int = 5,
    n_steps: int = 400,
    lr: float = 0.03,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    sigma_floor: float = 1e-3,
    max_samples: int = 1000,
    seed: int = 42,
) -> list[float]:
    """Find sigmas whose theoretical bodyness profile best matches the data.

    Minimizes the Total Variation Distance between:
      - The empirical normalized k-order correlation profile of the data, and
      - The theoretical kernel bodyness distribution induced by the sigma mixture.

    Uses JAX Adam on log(sigma) with box constraints.

    Args:
        x_train: Training data, shape (m, n_qubits).
        n_sigmas: Number of sigma bandwidths to optimise.
        n_steps: Adam iterations.
        lr: Learning rate.
        sigma_min: Lower bound for sigma (auto-derived if None).
        sigma_max: Upper bound for sigma (auto-derived if None).
        sigma_floor: Absolute minimum sigma value.
        max_samples: Subsample x_train for efficiency.
        seed: Random seed.

    Returns:
        List of optimised sigma values sorted descending.
    """
    import jax
    import jax.numpy as jnp

    x_sub = _subsample_data(x_train, max_samples=max_samples, seed=seed)
    n_qubits = x_sub.shape[1]
    max_k = n_qubits

    # Target: normalized empirical k-order correlation profile
    raw_profile = _k_order_correlation_distribution(x_sub, max_k=max_k)
    total = float(raw_profile.sum())
    if total == 0:
        print("[sigma:fit_k_order] Warning: zero correlation profile, falling back to fourier")
        return compute_sigma_fourier(n_qubits, n_sigmas)
    target = jnp.asarray(raw_profile / total)

    # Initialize from raw k-order heuristic
    init_sigmas = compute_sigma_k_order(
        x_sub, n_sigmas=n_sigmas, mode="raw",
        max_k=n_qubits // 2, sigma_floor=sigma_floor,
        max_samples=max_samples, seed=seed,
    )
    rng = np.random.default_rng(seed)
    while len(init_sigmas) < n_sigmas:
        s_min = sigma_floor if sigma_min is None else sigma_min
        s_max = max(init_sigmas) * 1.5 if sigma_max is None else sigma_max
        init_sigmas.append(float(np.exp(rng.uniform(np.log(s_min), np.log(s_max)))))

    # Auto-derive bounds from heuristic sigmas if not provided
    all_heuristic = list(init_sigmas)
    fourier_s = compute_sigma_fourier(n_qubits, n_sigmas)
    all_heuristic.extend(fourier_s)
    if sigma_min is None:
        sigma_min = max(sigma_floor, min(all_heuristic) * 0.5)
    if sigma_max is None:
        sigma_max = max(all_heuristic) * 1.5

    lower = jnp.log(jnp.asarray(sigma_min))
    upper = jnp.log(jnp.asarray(sigma_max))
    log_sigmas = jnp.clip(
        jnp.log(jnp.asarray(init_sigmas[:n_sigmas], dtype=jnp.float32)),
        lower, upper,
    )

    # Differentiable predicted bodyness distribution
    def _kernel_bodyness(log_s, n, mk):
        support = jnp.arange(1, mk + 1, dtype=log_s.dtype)[:, None]
        n_val = jnp.asarray(n, dtype=log_s.dtype)
        sigmas = jnp.exp(log_s)[None, :]
        probs = (1.0 - jnp.exp(-1.0 / (2.0 * sigmas**2))) / 2.0

        log_binom = (
            jax.scipy.special.gammaln(n_val + 1.0)
            - jax.scipy.special.gammaln(support + 1.0)
            - jax.scipy.special.gammaln(n_val - support + 1.0)
        )
        log_probs = (log_binom
                     + support * jnp.log(probs)
                     + (n_val - support) * jnp.log1p(-probs))
        mixture = jnp.mean(jnp.exp(log_probs), axis=1)
        return mixture / jnp.sum(mixture)

    def objective(log_s):
        predicted = _kernel_bodyness(log_s, n_qubits, max_k)
        return 0.5 * jnp.sum(jnp.abs(target - predicted))

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    # Adam optimization
    m_adam = jnp.zeros_like(log_sigmas)
    v_adam = jnp.zeros_like(log_sigmas)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    best_log_sigmas = log_sigmas
    best_value = float(objective(log_sigmas))

    for step in range(n_steps):
        value, grad = value_and_grad(log_sigmas)
        m_adam = beta1 * m_adam + (1.0 - beta1) * grad
        v_adam = beta2 * v_adam + (1.0 - beta2) * grad**2
        m_hat = m_adam / (1.0 - beta1 ** (step + 1))
        v_hat = v_adam / (1.0 - beta2 ** (step + 1))
        log_sigmas = jnp.clip(
            log_sigmas - lr * m_hat / (jnp.sqrt(v_hat) + adam_eps),
            lower, upper,
        )
        current_value = float(value)
        if current_value < best_value:
            best_value = current_value
            best_log_sigmas = log_sigmas

    final_sigmas = sorted([float(v) for v in jnp.exp(best_log_sigmas)], reverse=True)
    print(f"[sigma:fit_k_order] TVD={best_value:.6f}, "
          f"sigmas={[round(s, 6) for s in final_sigmas]}")

    return final_sigmas


# ---------------------------------------------------------------------------
# 8. data_driven_isotropic_pmf -- Empirical PMF from data
# ---------------------------------------------------------------------------

def compute_sigma_data_driven_isotropic_pmf(
    x_train: np.ndarray,
    mode: str = "pairwise",
    max_samples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Computes an arbitrary PMF over Walsh operator weights from the data's bodyness.

    Instead of fitting a Gaussian bandwidth (sigma), this outputs a fully 
    non-parametric 1D probability distribution Q(k) matching the target 
    manifold's correlation structure in the parity (Walsh) basis.

    Modes:
    - "pairwise": Exact spectral power. Squares individual parities then sums.
                  Captures localized structures and anti-correlations.
    - "mean_field": Averages row parities then squares. Assumes perfect 
                    global permutation symmetry in the data.

    Args:
        x_train: Training data, shape (m, n_qubits).
        mode: ``"pairwise"`` or ``"mean_field"``.
        max_samples: Subsample x_train for efficiency.
        seed: Random seed.

    Returns:
        A 1D numpy array of length `n_qubits + 1` representing the PMF,
        where pmf[k] is the probability of sampling an operator of weight k.
    """
    from src.utils import krawtchouk
    from math import comb
    from scipy.spatial.distance import cdist

    x_sub = _subsample_data(x_train, max_samples=max_samples, seed=seed)
    N, n_qubits = x_sub.shape

    pmf = np.zeros(n_qubits + 1, dtype=np.float64)

    if mode == "mean_field":
        # 1. Extract the row-weight histogram (Pr(w)) from x_sub
        weights = np.sum(x_sub, axis=1).astype(int)
        hist = np.bincount(weights, minlength=n_qubits + 1)
        pr_w = hist / N

        # 2. Compute the Krawtchouk transform to get average Pauli-Z moments
        for k in range(1, n_qubits + 1):
            z_k_expected = 0.0
            for w in range(n_qubits + 1):
                if pr_w[w] > 0:
                    krawtchouk_val = krawtchouk(k, w, n_qubits)
                    z_k_expected += pr_w[w] * (krawtchouk_val / comb(n_qubits, k))
                    
            # 3. Square the averaged energy for the PMF
            pmf[k] = z_k_expected ** 2

    elif mode == "pairwise":
        # 1. Compute exact Pairwise Hamming Distances natively
        # metric='cityblock' on 0/1 arrays gives integer Hamming distance.
        dists = cdist(x_sub, x_sub, metric='cityblock').flatten().astype(int)
        hist = np.bincount(dists, minlength=n_qubits + 1)
        pr_d = hist / (N * N)

        # 2. Exact analytic sum over all (n choose k) squared parities
        for k in range(1, n_qubits + 1):
            shell_energy = 0.0
            for d in range(n_qubits + 1):
                if pr_d[d] > 0:
                    shell_energy += pr_d[d] * krawtchouk(k, d, n_qubits)
            
            # Clip at 0 to avoid microscopic floating-point noise on empty shells
            pmf[k] = max(shell_energy, 0.0)
            
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'pairwise' or 'mean_field'.")

    # Normalize
    total = pmf.sum()
    if total > 0:
        pmf = pmf / total
    else:
        print("[sigma:data_driven_pmf] Warning: zero correlation profile, falling back to uniform")
        pmf[1:] = 1.0 / n_qubits

    print(f"[sigma:data_driven_pmf] Generated isotropic PMF (mode={mode})")
    return pmf


def compute_sigma_optimized(
    x_train: np.ndarray,
    n_sigmas: int = 5,
    n_ref: int = 1000,
    n_steps: int = 500,
    lr: float = 0.01,
    diversity_weight: float = 0.1,
    merge_threshold: float = 0.1,
    bernoulli_noise_p: float = 0.5,
    max_samples: int = 1000,
    seed: int = 42,
) -> list[float]:
    """Optimize sigma bandwidths by maximizing MMD test power (SNR).

    Finds the sigmas that maximize MMD^2 / sqrt(Var[MMD^2]) between the
    training data and a noisy reference, following the approach
    of Sutherland et al. (2017, "Generative Models and Model Criticism
    via Optimized Maximum Mean Discrepancy").

    The optimization is entirely classical (no circuit evaluation):
      1. Precompute Hamming distance matrices between data and reference
      2. Express kernel matrices K = exp(-H / (2*sigma^2)) as smooth
         functions of log(sigma)
      3. Compute unbiased MMD^2 and its variance from the kernel matrices
      4. Maximize the ratio (SNR) via JAX gradient ascent on log(sigma)

    A diversity regularizer prevents all sigmas from collapsing to the
    same value. After optimization, sigmas closer than ``merge_threshold``
    in log-space are merged (keeping the one with highest per-sigma SNR).

    Args:
        x_train: Training data, shape (m, n_qubits).
        n_sigmas: Number of sigma bandwidths to optimize.
        n_ref: Number of reference samples.
        n_steps: Gradient ascent iterations.
        lr: Learning rate for Adam optimizer.
        diversity_weight: Strength of the log-space repulsion regularizer.
        merge_threshold: Minimum distance in log(sigma) space between
            distinct sigmas. Sigmas closer than this are merged.
            Set to 0 to disable merging.
        bernoulli_noise_p: Probability of flipping each bit in x_train to
            create the reference distribution. p=0.5 gives uniform noise
            (recovers previous behavior), p=0.0 uses x_train as-is,
            p in (0, 1) gives noisy data. Default is 0.5.
        max_samples: Subsample x_train to this many rows.
        seed: Random seed.

    Returns:
        List of optimized sigma values (possibly fewer than n_sigmas
        if merging occurred), sorted descending.
    """
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)

    # --- Subsample data ---
    x_sub = _subsample_data(x_train, max_samples=max_samples, seed=seed)
    m, n_qubits = x_sub.shape

    # --- Generate reference with Bernoulli noise applied to training data ---
    # With p=0.5, each bit flips with 50% probability -> uniform distribution
    # With p=0.0, x_ref = x_sub (no noise)
    # With p in (0, 1), x_ref is x_sub with bit-flip noise
    x_ref_indices = rng.choice(m, size=min(n_ref, m * 2), replace=True)
    x_ref_base = x_sub[x_ref_indices % m]
    noise_mask = rng.binomial(1, bernoulli_noise_p, size=x_ref_base.shape)
    x_ref = ((x_ref_base.astype(int) + noise_mask) % 2).astype(np.float64)
    m_ref = len(x_ref)

    # Use equal-sized sets for the variance estimator (requires m == n)
    m_use = min(m, m_ref)
    x_d = x_sub[:m_use]
    x_r = x_ref[:m_use]

    # --- Precompute Hamming distance matrices (fixed, not differentiated) ---
    def hamming_matrix(a, b):
        return np.sum(a[:, None, :] != b[None, :, :], axis=2).astype(np.float64)

    H_dd = jnp.array(hamming_matrix(x_d, x_d))
    H_rr = jnp.array(hamming_matrix(x_r, x_r))
    H_dr = jnp.array(hamming_matrix(x_d, x_r))
    m_jnp = jnp.float64(m_use)

    # --- Initialize from fourier targets ---
    init_sigmas = compute_sigma_fourier(n_qubits, n_sigmas)
    if not init_sigmas:
        init_sigmas = [1.0] * n_sigmas
    log_sigmas_init = jnp.array([jnp.log(jnp.float64(s)) for s in init_sigmas])

    print(f"[sigma:optimized] Init from fourier: {[round(s, 6) for s in init_sigmas]}")

    # --- Define the SNR objective (to maximize) ---
    _eps = jnp.float64(1e-10)

    def _kernel_matrices(H, log_sigma):
        sigma2 = 2.0 * jnp.exp(2.0 * log_sigma)
        return jnp.exp(-H / sigma2)

    def _mmd2_and_variance_unbiased(K_XX, K_XY, K_YY, m_val):
        """Unbiased MMD^2 and variance (Gretton 2012 / Sutherland 2017)."""
        # Diagonal-free sums
        diag_X = jnp.ones(int(m_val))
        diag_Y = jnp.ones(int(m_val))
        sum_diag_X = m_val
        sum_diag_Y = m_val

        Kt_XX_sums = K_XX.sum(axis=1) - diag_X
        Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
        K_XY_sums_0 = K_XY.sum(axis=0)
        K_XY_sums_1 = K_XY.sum(axis=1)

        Kt_XX_sum = Kt_XX_sums.sum()
        Kt_YY_sum = Kt_YY_sums.sum()
        K_XY_sum = K_XY_sums_0.sum()

        Kt_XX_2_sum = (K_XX ** 2).sum() - m_val  # unit diagonal
        Kt_YY_2_sum = (K_YY ** 2).sum() - m_val
        K_XY_2_sum = (K_XY ** 2).sum()

        # Unbiased MMD^2
        mmd2 = (Kt_XX_sum / (m_val * (m_val - 1))
                + Kt_YY_sum / (m_val * (m_val - 1))
                - 2 * K_XY_sum / (m_val * m_val))

        # Variance of the U-statistic estimator
        var_est = (
            2 / (m_val**2 * (m_val - 1)**2) * (
                2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum
                + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
            - (4 * m_val - 6) / (m_val**3 * (m_val - 1)**3) * (
                Kt_XX_sum**2 + Kt_YY_sum**2)
            + 4 * (m_val - 2) / (m_val**3 * (m_val - 1)**2) * (
                K_XY_sums_1.dot(K_XY_sums_1)
                + K_XY_sums_0.dot(K_XY_sums_0))
            - 4 * (m_val - 3) / (m_val**3 * (m_val - 1)**2) * K_XY_2_sum
            - (8 * m_val - 12) / (m_val**5 * (m_val - 1)) * K_XY_sum**2
            + 8 / (m_val**3 * (m_val - 1)) * (
                1 / m_val * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
                - Kt_XX_sums.dot(K_XY_sums_1)
                - Kt_YY_sums.dot(K_XY_sums_0))
        )
        return mmd2, var_est

    def snr_objective(log_sigmas):
        """Total SNR across all sigmas + diversity regularizer."""
        total_snr = jnp.float64(0.0)
        for i in range(n_sigmas):
            ls = log_sigmas[i]
            K_dd_s = _kernel_matrices(H_dd, ls)
            K_rr_s = _kernel_matrices(H_rr, ls)
            K_dr_s = _kernel_matrices(H_dr, ls)
            mmd2, var = _mmd2_and_variance_unbiased(K_dd_s, K_dr_s, K_rr_s, m_jnp)
            snr = mmd2 / jnp.sqrt(jnp.maximum(var, _eps))
            total_snr = total_snr + snr

        # Diversity: Gaussian repulsion in log-space
        diversity = jnp.float64(0.0)
        for i in range(n_sigmas):
            for j in range(i + 1, n_sigmas):
                diversity = diversity + jnp.exp(
                    -(log_sigmas[i] - log_sigmas[j])**2)

        return total_snr - diversity_weight * diversity

    # --- Gradient ascent with Adam ---
    grad_fn = jax.jit(jax.grad(snr_objective))
    snr_fn = jax.jit(snr_objective)

    log_sigmas = jnp.array(log_sigmas_init, dtype=jnp.float64)

    # Adam state
    m_adam = jnp.zeros_like(log_sigmas)
    v_adam = jnp.zeros_like(log_sigmas)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    best_snr = float('-inf')
    best_log_sigmas = log_sigmas

    for step in range(n_steps):
        g = grad_fn(log_sigmas)
        m_adam = beta1 * m_adam + (1 - beta1) * g
        v_adam = beta2 * v_adam + (1 - beta2) * g**2
        m_hat = m_adam / (1 - beta1**(step + 1))
        v_hat = v_adam / (1 - beta2**(step + 1))
        log_sigmas = log_sigmas + lr * m_hat / (jnp.sqrt(v_hat) + adam_eps)

        if (step + 1) % 50 == 0 or step == 0:
            current_snr = float(snr_fn(log_sigmas))
            current_sigmas = [float(jnp.exp(ls)) for ls in log_sigmas]
            print(f"  step {step+1:4d}: SNR={current_snr:.4f}, "
                  f"sigmas={[round(s, 6) for s in sorted(current_sigmas, reverse=True)]}")
            if current_snr > best_snr:
                best_snr = current_snr
                best_log_sigmas = log_sigmas

    # --- Extract and deduplicate ---
    raw_sigmas = sorted([float(jnp.exp(ls)) for ls in best_log_sigmas], reverse=True)
    print(f"[sigma:optimized] Raw sigmas (SNR={best_snr:.4f}): "
          f"{[round(s, 6) for s in raw_sigmas]}")

    if merge_threshold > 0 and len(raw_sigmas) > 1:
        # Compute per-sigma SNR for ranking
        per_snr = []
        for ls in best_log_sigmas:
            K_dd_s = _kernel_matrices(H_dd, ls)
            K_rr_s = _kernel_matrices(H_rr, ls)
            K_dr_s = _kernel_matrices(H_dr, ls)
            mmd2, var = _mmd2_and_variance_unbiased(K_dd_s, K_dr_s, K_rr_s, m_jnp)
            per_snr.append(float(mmd2 / jnp.sqrt(jnp.maximum(var, _eps))))

        # Sort by log(sigma) descending, keep track of SNR
        indexed = sorted(enumerate(best_log_sigmas), key=lambda x: -float(x[1]))
        sorted_log = [float(best_log_sigmas[i]) for i, _ in indexed]
        sorted_snr = [per_snr[i] for i, _ in indexed]

        # Greedy merge: walk through sorted sigmas, skip if too close
        # to an already-kept sigma
        kept_log = [sorted_log[0]]
        kept_snr = [sorted_snr[0]]
        for k in range(1, len(sorted_log)):
            too_close = False
            for kl in kept_log:
                if abs(sorted_log[k] - kl) < merge_threshold:
                    too_close = True
                    break
            if not too_close:
                kept_log.append(sorted_log[k])
                kept_snr.append(sorted_snr[k])

        final_sigmas = sorted([float(np.exp(ls)) for ls in kept_log], reverse=True)
        n_merged = len(raw_sigmas) - len(final_sigmas)
        if n_merged > 0:
            print(f"[sigma:optimized] Merged {n_merged} redundant sigma(s) "
                  f"(threshold={merge_threshold:.2f} in log-space)")
        print(f"[sigma:optimized] Final {len(final_sigmas)} sigmas: "
              f"{[round(s, 6) for s in final_sigmas]}")
    else:
        final_sigmas = raw_sigmas

    return final_sigmas


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
        
        if method == 'fourier':
            n = x_train.shape[1]  # number of qubits = dimensionality
            s = int(heuristic.get('n_sigmas', 5))
            return compute_sigma_fourier(n, s)

        if method == 'k_order':
            n_sigmas = int(heuristic.get('n_sigmas', 5))
            mode = str(heuristic.get('mode', 'enriched'))
            max_k = heuristic.get('max_k')
            if max_k is not None:
                max_k = int(max_k)
            sigma_floor = float(heuristic.get('sigma_floor', 1e-3))
            return compute_sigma_k_order(
                x_train, n_sigmas=n_sigmas, mode=mode, max_k=max_k,
                sigma_floor=sigma_floor, max_samples=max_samples, seed=seed)

        if method == 'fit_k_order':
            n_sigmas = int(heuristic.get('n_sigmas', 5))
            n_steps = int(heuristic.get('n_steps', 400))
            opt_lr = float(heuristic.get('lr', 0.03))
            sigma_floor = float(heuristic.get('sigma_floor', 1e-3))
            s_min = heuristic.get('sigma_min')
            s_max = heuristic.get('sigma_max')
            if s_min is not None:
                s_min = float(s_min)
            if s_max is not None:
                s_max = float(s_max)
            return compute_sigma_fit_k_order(
                x_train, n_sigmas=n_sigmas, n_steps=n_steps, lr=opt_lr,
                sigma_min=s_min, sigma_max=s_max, sigma_floor=sigma_floor,
                max_samples=max_samples, seed=seed)

        if method == 'data_driven_isotropic_pmf':
            mode = str(heuristic.get('mode', 'pairwise'))
            return compute_sigma_data_driven_isotropic_pmf(
                x_train, mode=mode, max_samples=max_samples, seed=seed)

        if method == 'optimized':
            n_sigmas = int(heuristic.get('n_sigmas', 5))
            n_ref = int(heuristic.get('n_ref', 1000))
            n_steps = int(heuristic.get('n_steps', 400))
            opt_lr = float(heuristic.get('lr', 0.03))
            div_weight = float(heuristic.get('diversity_weight', 0.1))
            merge_thr = float(heuristic.get('merge_threshold', 0.1))
            bernoulli_p = float(heuristic.get('bernoulli_noise_p', 0.5))
            return compute_sigma_optimized(
                x_train, n_sigmas=n_sigmas, n_ref=n_ref,
                n_steps=n_steps, lr=opt_lr, diversity_weight=div_weight,
                merge_threshold=merge_thr, bernoulli_noise_p=bernoulli_p,
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
