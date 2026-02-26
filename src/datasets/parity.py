"""Parity dataset."""

import numpy as np
import matplotlib.pyplot as plt
from .base import BinaryDataset


class ParityDataset(BinaryDataset):
    """
    Generates even parity bitstrings.
    
    Each sample is a binary string where the sum of bits is even.
    """

    def __init__(self, n_qubits: int = 5):
        """
        Initialize the Parity dataset generator.
        
        Args:
            n_qubits: Number of qubits (bits) in each sample
        """
        super().__init__()
        self.n_qubits = n_qubits

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """
        Generate even parity samples.
        
        Generates n-1 random bits and sets the last bit to enforce even parity.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_samples, n_qubits) with binary values
        """
        rng = np.random.default_rng(seed)
        
        # Generate n-1 random bits
        samples = rng.integers(0, 2, size=(n_samples, self.n_qubits - 1))
        
        # Add final bit to enforce even parity
        parity_bits = (samples.sum(axis=1) % 2).reshape(-1, 1)
        samples = np.concatenate([samples, parity_bits], axis=1)
        
        self.data = np.array(samples, dtype=np.int8)
        return self.data

    def is_valid(self, sample: np.ndarray) -> bool:
        """Check if a sample has even parity."""
        return sample.sum() % 2 == 0

    def validity_rate(self, samples: np.ndarray) -> float:
        """Compute fraction of samples that have even parity."""
        valid = sum(self.is_valid(s) for s in samples)
        return valid / len(samples)

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        """
        Compute fraction of unique even parity strings represented in samples.
        
        Args:
            ground_truth: Not used for parity (coverage computed from theoretical space)
            samples: Model-generated samples
            
        Returns:
            Coverage rate (fraction of possible even parity strings in samples)
        """
        n_possible_strings = 2 ** (self.n_qubits - 1)
        
        # Get unique valid parity strings from samples
        parity_strings = set()
        for s in samples:
            if self.is_valid(s):
                parity_strings.add(tuple(s))
        
        return len(parity_strings) / n_possible_strings

    def visualize(self, sample: np.ndarray, ax=None):
        """
        Visualize a parity sample as a binary string.
        
        Args:
            sample: 1D binary array
            ax: Matplotlib axis to plot on (creates new if None)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.n_qubits * 0.5 + 1, 2))
        
        # Display as a row of colored cells
        sample_2d = sample.reshape(1, -1)
        ax.imshow(sample_2d, cmap='binary', interpolation='nearest', aspect='auto')
        
        # Add bit values as text
        for i, bit in enumerate(sample):
            ax.text(i, 0, str(int(bit)), ha='center', va='center', 
                   color='red' if bit == 0 else 'white', fontsize=12, fontweight='bold')
        
        # Add grid lines
        for i in range(self.n_qubits + 1):
            ax.axvline(i - 0.5, color='gray', linewidth=1)
        ax.axhline(-0.5, color='gray', linewidth=1)
        ax.axhline(0.5, color='gray', linewidth=1)
        
        ax.set_xticks(range(self.n_qubits))
        ax.set_xticklabels([f'q{i}' for i in range(self.n_qubits)])
        ax.set_yticks([])
        
        # Title with parity information
        parity_sum = sample.sum()
        is_even = parity_sum % 2 == 0
        ax.set_title(f"Sum: {parity_sum} ({'EVEN' if is_even else 'ODD'} parity)", 
                    fontweight='bold', 
                    color='green' if is_even else 'red')
        
        return ax
