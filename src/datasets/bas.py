"""Bars and Stripes (BAS) dataset."""

import numpy as np
import matplotlib.pyplot as plt
from .base import BinaryDataset


class BarsAndStripesDataset(BinaryDataset):
    """
    Generates Bars and Stripes (BAS) patterns.
    
    Each sample is a binary grid that is either:
    - A bar: All active rows are identical across a contiguous vertical extent
    - A stripe: All active columns are identical across a contiguous horizontal extent

    By default, bars and stripes span the full grid, matching the canonical
    BAS dataset. Set ``length`` or the axis-specific lengths to generate
    clean partial bars/stripes inside a larger grid.
    
    Valid patterns are exactly those that satisfy one of these constraints.
    """

    def __init__(
        self,
        height: int = 3,
        width: int = 3,
        length: int | None = None,
        bar_length: int | None = None,
        stripe_length: int | None = None,
        min_spacing: int = 0,
    ):
        """
        Initialize the BAS dataset generator.
        
        Args:
            height: Height of the grid (number of rows)
            width: Width of the grid (number of columns)
            length: Shared contiguous extent for bars and stripes. If omitted,
                bars and stripes span the full grid.
            bar_length: Contiguous vertical extent for bar samples.
            stripe_length: Contiguous horizontal extent for stripe samples.
            min_spacing: Minimum number of inactive cells required between
                active bars or stripes along the varying axis. The default
                ``0`` preserves canonical BAS; ``1`` forbids side-by-side
                active bars/stripes.
        """
        super().__init__()
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if min_spacing < 0:
            raise ValueError("min_spacing must be non-negative")
        self.height = int(height)
        self.width = int(width)
        self.bar_length = self._resolve_length(bar_length, length, self.height, "bar_length")
        self.stripe_length = self._resolve_length(stripe_length, length, self.width, "stripe_length")
        self.min_spacing = int(min_spacing)
        self.n_qubits = self.height * self.width
        self._bar_patterns = self._spaced_binary_patterns(self.width)
        self._stripe_patterns = self._spaced_binary_patterns(self.height)
        
        # Pre-compute all unique valid bar and stripe patterns
        self._valid_patterns = self._compute_valid_patterns()

    @staticmethod
    def _resolve_length(
        axis_length: int | None,
        shared_length: int | None,
        limit: int,
        name: str,
    ) -> int:
        value = axis_length if axis_length is not None else shared_length
        if value is None:
            return int(limit)
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        if value > limit:
            raise ValueError(f"{name} must be less than or equal to the grid extent ({limit})")
        return value

    def _spaced_binary_patterns(self, size: int) -> np.ndarray:
        patterns = []
        for pattern in range(2**size):
            bits = np.array([(pattern >> i) & 1 for i in range(size)], dtype=np.int8)
            active = np.flatnonzero(bits)
            if len(active) < 2 or np.all(np.diff(active) > self.min_spacing):
                patterns.append(bits)
        return np.asarray(patterns, dtype=np.int8)

    def _compute_valid_patterns(self) -> set:
        """
        Enumerate all unique valid BAS patterns (bars and stripes).
        
        Returns:
            Set of tuples representing all valid patterns
        """
        patterns = set()
        
        # All possible bars (active rows are identical over a contiguous extent)
        for row_start in range(0, self.height - self.bar_length + 1):
            for row in self._bar_patterns:
                grid = np.zeros((self.height, self.width), dtype=np.int8)
                grid[row_start:row_start + self.bar_length, :] = row
                patterns.add(tuple(grid.flatten().astype(int)))
        
        # All possible stripes (active columns are identical over a contiguous extent)
        for col_start in range(0, self.width - self.stripe_length + 1):
            for col in self._stripe_patterns:
                grid = np.zeros((self.height, self.width), dtype=np.int8)
                grid[:, col_start:col_start + self.stripe_length] = col[:, None]
                patterns.add(tuple(grid.flatten().astype(int)))
        
        return patterns

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """
        Generate BAS samples.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_samples, height*width) with binary values
        """
        rng = np.random.default_rng(seed)
        samples = []
        
        for _ in range(n_samples):
            if rng.random() < 0.5:
                # Vertical bars: active rows are identical
                bars = self._bar_patterns[int(rng.integers(0, len(self._bar_patterns)))]
                row_start = int(rng.integers(0, self.height - self.bar_length + 1))
                img = np.zeros((self.height, self.width), dtype=np.int8)
                img[row_start:row_start + self.bar_length, :] = bars
            else:
                # Horizontal stripes: active columns are identical
                stripes = self._stripe_patterns[int(rng.integers(0, len(self._stripe_patterns)))]
                col_start = int(rng.integers(0, self.width - self.stripe_length + 1))
                img = np.zeros((self.height, self.width), dtype=np.int8)
                img[:, col_start:col_start + self.stripe_length] = stripes[:, None]
            
            samples.append(img.reshape(-1))
        
        self.data = np.array(samples, dtype=np.int8)
        return self.data



    def visualize(self, sample: np.ndarray, ax=None):
        """
        Visualize a BAS sample as a 2D grid.
        
        Args:
            sample: 1D binary array of length height*width
            ax: Matplotlib axis to plot on (creates new if None)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        
        img = sample.reshape(self.height, self.width)
        ax.imshow(img, cmap='binary', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid lines
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)
        
        if self.is_valid(sample):
            length_label = ""
            if self.bar_length != self.height or self.stripe_length != self.width:
                length_label = f" (bar={self.bar_length}, stripe={self.stripe_length})"
            if self.min_spacing:
                length_label += f" spacing={self.min_spacing}"
            label = f"Valid BAS{length_label}"
            ax.set_title(label)
        else:
            ax.set_title("Invalid BAS pattern")
        
        return ax
