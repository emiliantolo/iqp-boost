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
