"""Gaussian mixture dataset."""

import numpy as np
import matplotlib.pyplot as plt
from .base import BinaryDataset


class GaussianMixtureDataset(BinaryDataset):
    """
    Generates samples from a grid of 2D isotropic Gaussians.
    
    The continuous 2D samples are discretized to binary representation
    for use in quantum circuits.
    """

    def __init__(self, grid_size: int = 3, spread: float = 0.3, 
                 separation: float = 3.0, grid_bits: int = 4):
        """
        Initialize the Gaussian mixture dataset generator.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size Gaussians)
            spread: Standard deviation of each Gaussian
            separation: Distance between adjacent Gaussian centers
            grid_bits: Bits per dimension for binary encoding (total qubits = 2*grid_bits)
        """
        super().__init__()
        self.grid_size = grid_size
        self.spread = spread
        self.separation = separation
        self.grid_bits = grid_bits
        self.n_qubits = 2 * grid_bits
        
        # Create grid of Gaussian centers
        self.centers = []
        for i in range(grid_size):
            for j in range(grid_size):
                center = np.array([i * separation, j * separation])
                self.centers.append(center)
        self.centers = np.array(self.centers)
        
        # Normalization bounds (set during discretization)
        self._norm_bounds = None

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """
        Generate Gaussian mixture samples.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_samples, 2*grid_bits) with binary values
        """
        rng = np.random.default_rng(seed)
        
        # Sample uniformly from each Gaussian
        n_gaussians = len(self.centers)
        samples_per_gaussian = n_samples // n_gaussians
        remainder = n_samples % n_gaussians
        
        samples_2d = []
        for idx, center in enumerate(self.centers):
            k = samples_per_gaussian + (1 if idx < remainder else 0)
            cluster_samples = rng.normal(center, self.spread, size=(k, 2))
            samples_2d.append(cluster_samples)
        
        samples_2d = np.vstack(samples_2d)
        perm = rng.permutation(len(samples_2d))
        samples_2d = samples_2d[perm]
        
        # Store continuous data for reference
        self.data_continuous = samples_2d
        
        # Discretize to binary
        self.data = self._discretize_to_binary(samples_2d)
        return self.data

    def _discretize_to_binary(self, samples_2d: np.ndarray) -> np.ndarray:
        """
        Discretize 2D continuous samples to binary representation.
        
        Args:
            samples_2d: (n_samples, 2) array of 2D points
            
        Returns:
            (n_samples, 2*grid_bits) binary array
        """
        # Normalize to [0, 1] range
        x_min, x_max = samples_2d[:, 0].min(), samples_2d[:, 0].max()
        y_min, y_max = samples_2d[:, 1].min(), samples_2d[:, 1].max()
        
        # Store normalization bounds for inverse mapping
        self._norm_bounds = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
        
        x_norm = (samples_2d[:, 0] - x_min) / (x_max - x_min + 1e-8)
        y_norm = (samples_2d[:, 1] - y_min) / (y_max - y_min + 1e-8)
        
        # Discretize each dimension to grid_bits bits
        max_idx = 2**self.grid_bits - 1
        x_idx = np.clip(np.round(x_norm * max_idx), 0, max_idx).astype(int)
        y_idx = np.clip(np.round(y_norm * max_idx), 0, max_idx).astype(int)
        
        # Convert to binary representation
        x_bits = np.vstack([[(x >> i) & 1 for i in range(self.grid_bits)] for x in x_idx])
        y_bits = np.vstack([[(y >> i) & 1 for i in range(self.grid_bits)] for y in y_idx])
        
        return np.hstack([x_bits, y_bits]).astype(np.int8)

    def binary_to_continuous(self, binary_samples: np.ndarray) -> np.ndarray:
        """
        Decode binary samples back to 2D continuous coordinates.
        
        Args:
            binary_samples: (n_samples, 2*grid_bits) binary array
            
        Returns:
            (n_samples, 2) array of 2D coordinates
        """
        if self._norm_bounds is None:
            raise ValueError("Dataset must be generated first to establish normalization bounds")
        
        # Extract x and y bits
        x_bits = binary_samples[:, :self.grid_bits]
        y_bits = binary_samples[:, self.grid_bits:]
        
        # Convert from binary to integer indices
        x_idx = np.sum(x_bits * (2 ** np.arange(self.grid_bits)), axis=1)
        y_idx = np.sum(y_bits * (2 ** np.arange(self.grid_bits)), axis=1)
        
        # Map to [0, 1] range
        max_idx = 2**self.grid_bits - 1
        x_norm = x_idx / max_idx
        y_norm = y_idx / max_idx
        
        # Rescale to original coordinate system
        x_range = self._norm_bounds['x_max'] - self._norm_bounds['x_min'] + 1e-8
        y_range = self._norm_bounds['y_max'] - self._norm_bounds['y_min'] + 1e-8
        x_data = x_norm * x_range + self._norm_bounds['x_min']
        y_data = y_norm * y_range + self._norm_bounds['y_min']
        
        return np.column_stack([x_data, y_data])

    def validity_rate(self, samples: np.ndarray, threshold_std: float = 2.0) -> float:
        """
        Measure how many samples are close to one of the true cluster centers.
        
        Args:
            samples: Binary samples to validate
            threshold_std: Distance threshold in standard deviations
            
        Returns:
            Fraction of samples within threshold of any cluster center
        """
        # Convert binary samples to continuous
        samples_2d = self.binary_to_continuous(samples)
        
        # Compute distances to all centers
        distances = np.linalg.norm(samples_2d[:, None, :] - self.centers[None, :, :], axis=2)
        nearest_dist = np.min(distances, axis=1)
        
        threshold = threshold_std * self.spread
        valid = np.sum(nearest_dist < threshold)
        return valid / len(samples)

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray, 
                     threshold_std: float = 2.0) -> float:
        """
        Measure what fraction of the true Gaussian clusters are well-represented.
        
        Args:
            ground_truth: Not used (coverage computed from true centers)
            samples: Binary samples to evaluate
            threshold_std: Distance threshold in standard deviations
            
        Returns:
            Fraction of cluster centers that have samples nearby
        """
        # Convert binary samples to continuous
        samples_2d = self.binary_to_continuous(samples)
        
        threshold = threshold_std * self.spread
        covered = 0
        
        for center in self.centers:
            distances = np.linalg.norm(samples_2d - center[None, :], axis=1)
            if np.any(distances < threshold):
                covered += 1
        
        return covered / len(self.centers)

    def visualize(self, sample: np.ndarray, ax=None):
        """
        Visualize a Gaussian mixture sample in 2D space.
        
        Args:
            sample: 1D binary array (will be converted to 2D coordinates)
            ax: Matplotlib axis to plot on (creates new if None)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        # Convert binary sample to 2D coordinates
        sample_2d = self.binary_to_continuous(sample.reshape(1, -1))[0]
        
        # Plot cluster centers
        ax.scatter(self.centers[:, 0], self.centers[:, 1], 
                  s=200, c='red', marker='x', linewidths=3, 
                  label='Cluster centers', zorder=3)
        
        # Draw circles around centers (2-sigma radius)
        for center in self.centers:
            circle = plt.Circle(center, 2 * self.spread, 
                              color='red', fill=False, 
                              linestyle='--', alpha=0.3)
            ax.add_patch(circle)
        
        # Plot the sample
        ax.scatter(sample_2d[0], sample_2d[1], 
                  s=150, c='blue', marker='o', 
                  edgecolors='black', linewidth=2,
                  label='Sample', zorder=4)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Gaussian Mixture Sample')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        return ax
