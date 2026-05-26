"""Bars and Stripes (BAS) dataset."""

import numpy as np
import matplotlib.pyplot as plt
from .base import BinaryDataset


class BarsAndStripesDataset(BinaryDataset):
    """
    Generates Bars and Stripes (BAS) patterns.
    
    Each sample is a binary grid that is either:
    - A bar: All rows are identical
    - A stripe: All columns are identical
    
    Valid patterns are exactly those that satisfy one of these constraints.
    """

    def __init__(self, height: int = 3, width: int = 3):
        """
        Initialize the BAS dataset generator.
        
        Args:
            height: Height of the grid (number of rows)
            width: Width of the grid (number of columns)
        """
        super().__init__()
        self.height = height
        self.width = width
        self.n_qubits = height * width
        
        # Pre-compute all unique valid bar and stripe patterns
        self._valid_patterns = self._compute_valid_patterns()

    def _compute_valid_patterns(self) -> set:
        """
        Enumerate all unique valid BAS patterns (bars and stripes).
        
        Returns:
            Set of tuples representing all valid patterns
        """
        patterns = set()
        
        # All possible bars (all rows identical)
        for row_pattern in range(2**self.width):
            row = np.array([(row_pattern >> i) & 1 for i in range(self.width)], dtype=np.int8)
            grid = np.tile(row, (self.height, 1))
            patterns.add(tuple(grid.flatten().astype(int)))
        
        # All possible stripes (all columns identical)
        for col_pattern in range(2**self.height):
            col = np.array([(col_pattern >> i) & 1 for i in range(self.height)], dtype=np.int8)
            grid = np.tile(col[:, None], (1, self.width))
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
                # Vertical bars: All rows identical
                bars = rng.integers(0, 2, size=(self.width,))
                img = np.tile(bars, (self.height, 1))
            else:
                # Horizontal stripes: All columns identical
                stripes = rng.integers(0, 2, size=(self.height,))
                img = np.tile(stripes[:, None], (1, self.width))
            
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
        
        # Label as bar or stripe
        is_bar = self.is_valid(sample)
        img_2d = sample.reshape(self.height, self.width)
        is_vertical = all(np.array_equal(img_2d[0], row) for row in img_2d)
        
        if is_bar:
            label = "Bar (vertical)" if is_vertical else "Stripe (horizontal)"
            ax.set_title(label)
        else:
            ax.set_title("Invalid BAS pattern")
        
        return ax


class VariableLengthBarsAndStripesDataset(BinaryDataset):
    """
    Generate unions of variable-length contiguous bars and stripes.

    A segment is either a contiguous run of full rows or a contiguous run of
    full columns. Valid patterns are unions of one or more segments, up to
    ``max_segments`` total segments.
    """

    def __init__(
        self,
        height: int = 5,
        width: int = 5,
        min_length: int = 1,
        max_length: int | None = None,
        max_segments: int = 2,
    ):
        super().__init__()
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if min_length <= 0:
            raise ValueError("min_length must be positive")
        if max_segments <= 0:
            raise ValueError("max_segments must be positive")

        self.height = int(height)
        self.width = int(width)
        self.n_qubits = self.height * self.width
        self.min_length = int(min_length)
        self.max_length = int(max_length) if max_length is not None else max(self.height, self.width)
        self.max_segments = int(max_segments)
        if self.max_length < self.min_length:
            raise ValueError("max_length must be greater than or equal to min_length")

        self._valid_patterns = self._compute_valid_patterns()
        if not self._valid_patterns:
            raise ValueError("no valid variable-length BAS patterns for the requested lengths")
        self.support = np.asarray(sorted(self._valid_patterns), dtype=np.int8)
        self.support_size = int(len(self.support))

    def _segments(self) -> list[np.ndarray]:
        segments = []

        row_max = min(self.max_length, self.height)
        for length in range(self.min_length, row_max + 1):
            for start in range(0, self.height - length + 1):
                grid = np.zeros((self.height, self.width), dtype=np.int8)
                grid[start:start + length, :] = 1
                segments.append(grid)

        col_max = min(self.max_length, self.width)
        for length in range(self.min_length, col_max + 1):
            for start in range(0, self.width - length + 1):
                grid = np.zeros((self.height, self.width), dtype=np.int8)
                grid[:, start:start + length] = 1
                segments.append(grid)

        return segments

    def _compute_valid_patterns(self) -> set:
        from itertools import combinations

        segments = self._segments()
        patterns = set()
        max_segments = min(self.max_segments, len(segments))
        for segment_count in range(1, max_segments + 1):
            for chosen in combinations(range(len(segments)), segment_count):
                grid = np.zeros((self.height, self.width), dtype=np.int8)
                for index in chosen:
                    grid = np.maximum(grid, segments[index])
                patterns.add(tuple(grid.reshape(-1).astype(int)))
        return patterns

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.support_size, size=int(n_samples), replace=True)
        self.data = self.support[indices].astype(np.int8)
        return self.data

    def _validate_samples(self, samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=np.int8)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.n_qubits:
            raise ValueError(f"expected samples with shape (n, {self.n_qubits})")
        return (values > 0).astype(np.int8)

    def valid_count(self, samples: np.ndarray) -> int:
        values = self._validate_samples(samples)
        return int(sum(tuple(row.tolist()) in self._valid_patterns for row in values))

    def validity_rate(self, samples: np.ndarray) -> float:
        values = self._validate_samples(samples)
        if len(values) == 0:
            return 0.0
        return float(self.valid_count(values) / len(values))

    def unique_valid_count(self, samples: np.ndarray) -> int:
        values = self._validate_samples(samples)
        return int(len({tuple(row.tolist()) for row in values if tuple(row.tolist()) in self._valid_patterns}))

    def visualize(self, sample: np.ndarray, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))

        img = np.asarray(sample).reshape(self.height, self.width)
        ax.imshow(img, cmap='binary', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)
        ax.set_title(
            f"Variable BAS ({self.min_length}-{self.max_length}, up to {self.max_segments})"
        )
        return ax
