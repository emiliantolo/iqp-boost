import pickle
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path


class BinaryDataset(ABC):
    """
    Abstract base class for all binary dataset generators.
    Ensures that generated data is represented as binary strings (numpy arrays of 0s and 1s)
    and provides unified saving/loading functionality.
    """

    def __init__(self, data: np.ndarray = None):
        """
        Initialize the dataset.

        Args:
            data (np.ndarray, optional): The binary data matrix. Should be of dtype np.int8
                and contain only 0s and 1s.
        """
        self.data = np.asarray(data, dtype=np.int8) if data is not None else None

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
