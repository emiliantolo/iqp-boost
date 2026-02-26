"""Parity dataset."""

import numpy as np
import matplotlib.pyplot as plt
from .base import BinaryDataset


class ParityDataset(BinaryDataset):
    """
    Generates even parity bitstrings.
    
    Each sample is a binary string where the sum of bits is even.
    Valid patterns are all 2^(n_qubits-1) even-parity bitstrings.
    """

    def __init__(self, n_qubits: int = 5):
        """
        Initialize the Parity dataset generator.
        
        Args:
            n_qubits: Number of qubits (bits) in each sample
        """
        super().__init__()
        self.n_qubits = n_qubits
        
        # Pre-compute all valid even-parity patterns
        self._valid_patterns = self._enumerate_valid_patterns()

    def _enumerate_valid_patterns(self) -> set:
        """
        Enumerate all 2^(n_qubits-1) valid even-parity bitstrings.
        
        Returns:
            Set of tuples representing all even-parity patterns
        """
        patterns = set()
        n_combinations = 2 ** (self.n_qubits - 1)
        
        # Generate all combinations of first n_qubits-1 bits
        for i in range(n_combinations):
            bits = np.array([(i >> j) & 1 for j in range(self.n_qubits - 1)], dtype=np.int8)
            # Add parity bit to make total sum even
            parity_bit = bits.sum() % 2
            full_pattern = np.concatenate([bits, [parity_bit]])
            patterns.add(tuple(full_pattern.astype(int)))
        
        return patterns

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
