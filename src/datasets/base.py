import pickle
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path


class BinaryDataset(ABC):
    """
    Abstract base class for all binary dataset generators.
    Ensures that generated data is represented as binary strings (numpy arrays of 0s and 1s)
    and provides unified saving/loading functionality.
    
    Datasets can optionally implement pattern-space membership validation by:
    1. Setting self._valid_patterns to a set of tuples (valid patterns)
    2. Calling self._ensure_valid_patterns() to lazily populate (optional)
    """

    def __init__(self, data: np.ndarray = None):
        """
        Initialize the dataset.

        Args:
            data (np.ndarray, optional): The binary data matrix. Should be of dtype np.int8
                and contain only 0s and 1s.
        """
        self.data = np.asarray(data, dtype=np.int8) if data is not None else None
        self._valid_patterns = None  # Subclasses can populate this

    @abstractmethod
    def generate(self, *args, **kwargs) -> np.ndarray:
        """
        Generates the binary dataset.
        Must be implemented by subclasses.
        Should set self.data and return it.
        """
        pass

    @abstractmethod
    def visualize(self, sample: np.ndarray, ax=None):
        """
        Visualizes a single binary sample produced by the dataset or a model.
        
        Args:
            sample (np.ndarray): A 1D array of 0s and 1s representing a single sample.
            ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on.
                If None, a new figure and axis should be created.
        """
        pass

    def _ensure_valid_patterns(self):
        """
        Hook for subclasses to lazily compute/populate _valid_patterns.
        Override this if your dataset uses lazy evaluation of valid patterns.
        Default: does nothing (patterns should be pre-computed in __init__).
        """
        pass

    def is_valid(self, sample: np.ndarray) -> bool:
        """
        Check if a sample is a valid pattern (member of the pattern space).
        
        Requires: _valid_patterns must be a set of tuples.
        
        Args:
            sample: 1D binary array
            
        Returns:
            True if sample is in the valid pattern space
        """
        if self._valid_patterns is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set _valid_patterns "
                "to use is_valid(). Override is_valid() for custom logic."
            )
        self._ensure_valid_patterns()
        return tuple(sample.astype(int)) in self._valid_patterns

    def validity_rate(self, samples: np.ndarray) -> float:
        """
        Compute fraction of samples that are valid patterns.
        
        Requires: _valid_patterns must be a set of tuples.
        
        Args:
            samples (np.ndarray): Matrix of shape (n_samples, n_qubits)
            
        Returns:
            float: Fraction of valid samples
        """
        self._ensure_valid_patterns()  # Allow lazy loading
        if self._valid_patterns is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set _valid_patterns "
                "to use validity_rate(). Override validity_rate() for custom logic."
            )
        valid_count = sum(self.is_valid(s) for s in samples)
        return valid_count / len(samples) if len(samples) > 0 else 0.0

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        """
        Compute what fraction of all possible valid patterns appear in generated samples.
        
        Requires: _valid_patterns must be a set of tuples.
        
        Args:
            ground_truth: Not used (coverage computed from canonical valid_patterns)
            samples (np.ndarray): Generated samples of shape (n_samples, n_qubits)
            
        Returns:
            float: Fraction of valid patterns found in generated samples
        """
        self._ensure_valid_patterns()  # Allow lazy loading
        if self._valid_patterns is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set _valid_patterns "
                "to use coverage_rate(). Override coverage_rate() for custom logic."
            )
        if len(samples) == 0:
            return 0.0
        
        sample_set = set(map(tuple, samples))
        valid_generated = sample_set & self._valid_patterns
        
        return len(valid_generated) / len(self._valid_patterns)

    def save(self, filepath: str | Path):
        """
        Saves the dataset (the entire object) to a pickle file.

        Args:
            filepath (str | Path): Path to save the file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> "BinaryDataset":
        """
        Loads a dataset from a pickle file.

        Args:
            filepath (str | Path): Path to the pickle file.

        Returns:
            BinaryDataset: The loaded dataset instance.
        """
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise TypeError(f"Loaded object is of type {type(obj)}, expected {cls}")
            return obj

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, idx):
        if self.data is None:
            raise ValueError("Dataset strictly needs to be generated or loaded first.")
        return self.data[idx]

    @property
    def shape(self):
        return self.data.shape if self.data is not None else None
