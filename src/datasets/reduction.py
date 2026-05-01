"""Shared dimensionality-reduction utilities for large-scale binary datasets."""

import numpy as np
from sklearn.decomposition import PCA


def pca_reduce(data: np.ndarray, n_components: int, seed: int = 0) -> np.ndarray:
    """
    Reduce binary data to *n_components* features via sklearn PCA, then re-binarise.

    The PCA is fitted on the supplied *data* matrix.  After projection each
    feature is thresholded at its **median** value so that the output remains
    binary (0/1) while preserving as much variance structure as possible.

    Args:
        data: Binary array of shape ``(n_samples, n_features)`` with values in {0, 1}.
        n_components: Target number of output features (qubits).
        seed: Random seed forwarded to :class:`sklearn.decomposition.PCA`.

    Returns:
        Binary array of shape ``(n_samples, n_components)`` with dtype ``np.int8``.
    """
    if n_components >= data.shape[1]:
        return data.astype(np.int8)

    pca = PCA(n_components=n_components, random_state=seed)
    projected = pca.fit_transform(data.astype(np.float64))

    # Re-binarise: threshold each component at its median
    medians = np.median(projected, axis=0, keepdims=True)
    binarised = (projected > medians).astype(np.int8)
    return binarised


def umap_reduce(
    data: np.ndarray,
    n_components: int,
    seed: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> np.ndarray:
    """Reduce binary data to *n_components* features via UMAP, then re-binarise.

    UMAP is fitted on the supplied *data* matrix. After projection, each
    component is thresholded at its median value so the output remains binary.
    """
    if n_components >= data.shape[1]:
        return data.astype(np.int8)

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP reduction requires the 'umap-learn' package. Install it or use a different reduction mode."
        ) from exc

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    projected = reducer.fit_transform(data.astype(np.float64))

    medians = np.median(projected, axis=0, keepdims=True)
    binarised = (projected > medians).astype(np.int8)
    return binarised


def _binary_mutual_information_matrix(data: np.ndarray) -> np.ndarray:
    """Compute the pairwise mutual-information matrix for binary features."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("data must be a 2-D array")

    n_samples, n_features = x.shape
    if n_features == 0:
        return np.zeros((0, 0), dtype=np.float64)

    p1 = x.mean(axis=0)
    p0 = 1.0 - p1
    p11 = (x.T @ x) / float(n_samples)
    p10 = p1[:, None] - p11
    p01 = p1[None, :] - p11
    p00 = 1.0 - p11 - p10 - p01

    def _mi_term(prob_xy: np.ndarray, prob_x: np.ndarray, prob_y: np.ndarray) -> np.ndarray:
        denom = prob_x[:, None] * prob_y[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(prob_xy > 0.0, prob_xy * np.log(prob_xy / denom), 0.0)
        term[~np.isfinite(term)] = 0.0
        return term

    mi = (
        _mi_term(p11, p1, p1)
        + _mi_term(p10, p1, p0)
        + _mi_term(p01, p0, p1)
        + _mi_term(p00, p0, p0)
    )
    np.fill_diagonal(mi, 0.0)
    mi[mi < 0.0] = 0.0
    return mi


def variance_mi_reduce(
    data: np.ndarray,
    n_components: int,
    variance_filter_dims: int = 100,
) -> np.ndarray:
    """Select a binary feature subset using variance filtering followed by MI ranking.

    The reducer works directly at the qubit level, without projecting to a dense latent
    space. First it keeps the ``variance_filter_dims`` highest-variance features as a
    coarse pre-filter. It then computes the pairwise mutual-information matrix on that
    reduced set, scores each feature by its total MI with the rest of the retained
    features, and keeps the top ``n_components`` features.

    The selected features are returned in their original order, so the resulting qubit
    indices remain aligned with the source ordering even though the grid structure is no
    longer meaningful.
    """
    if n_components >= data.shape[1]:
        return data.astype(np.int8)

    if variance_filter_dims <= 0:
        raise ValueError("variance_filter_dims must be positive")

    x = np.asarray(data, dtype=np.int8)
    n_features = x.shape[1]
    intermediate_dims = min(max(int(variance_filter_dims), int(n_components)), n_features)

    variances = x.astype(np.float64).var(axis=0)
    variance_order = np.lexsort((np.arange(n_features), -variances))
    variance_keep = np.sort(variance_order[:intermediate_dims])
    coarse = x[:, variance_keep]

    mi_matrix = _binary_mutual_information_matrix(coarse)
    mi_scores = mi_matrix.sum(axis=1)
    mi_order = np.lexsort((np.arange(coarse.shape[1]), -mi_scores))
    selected_local = np.sort(mi_order[:n_components])

    return coarse[:, selected_local].astype(np.int8)


def spatial_downsample(
    data: np.ndarray,
    src_size: int,
    dst_height: int,
    dst_width: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Downsample 2-D binary images via average-pooling and re-binarisation.

    Each row of *data* is reshaped to ``(src_size, src_size)``, block-averaged
    to ``(dst_height, dst_width)``, and then thresholded.

    Args:
        data: Binary array of shape ``(n_samples, src_size**2)``.
        src_size: Side length of the original square image.
        dst_height: Height of the target image.
        dst_width: Width of the target image.
        threshold: Value above which a pooled pixel is set to 1.

    Returns:
        Binary array of shape ``(n_samples, dst_height * dst_width)`` with
        dtype ``np.int8``.
    """
    if dst_height >= src_size and dst_width >= src_size:
        return data.astype(np.int8)

    n_samples = data.shape[0]
    images = data.reshape(n_samples, src_size, src_size).astype(np.float64)

    block_h = src_size // dst_height
    block_w = src_size // dst_width

    # Trim to a size evenly divisible by destination dims
    trim_h = block_h * dst_height
    trim_w = block_w * dst_width
    images = images[:, :trim_h, :trim_w]

    # Reshape into blocks and average
    images = images.reshape(n_samples, dst_height, block_h, dst_width, block_w)
    pooled = images.mean(axis=(2, 4))

    binarised = (pooled > threshold).astype(np.int8)
    return binarised.reshape(n_samples, dst_height * dst_width)
