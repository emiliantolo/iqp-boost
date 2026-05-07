"""Graph isomorphism permutation dataset."""

from itertools import permutations
from math import factorial

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism

from .base import BinaryDataset


class GraphIsomorphismDataset(BinaryDataset):
    """
    Enumerates adjacency-matrix permutations of one seed graph.

    Each sample is the flattened upper-triangular part of an adjacency matrix
    after applying a node permutation to a single Erdős-Rényi seed graph.
    """

    def __init__(
        self,
        num_nodes: int = 8,
        edge_prob: float = 0.5,
        train_split_ratio: float = 0.125,
        seed: int = 42,
        enforce_asymmetry: bool = True,
    ):
        """
        Initialize the graph isomorphism dataset.

        Args:
            num_nodes: Number of graph nodes.
            edge_prob: Edge probability for the Erdős-Rényi seed graph.
            train_split_ratio: Fraction of unique permutations assigned to train.
            seed: Seed for graph generation and split shuffling.
            enforce_asymmetry: If true, reject seed graphs with automorphisms.
        """
        super().__init__()
        if num_nodes < 2:
            raise ValueError(f"num_nodes must be at least 2, got {num_nodes}")
        if enforce_asymmetry and num_nodes < 6:
            raise ValueError(
                "Asymmetric simple graphs require at least 6 nodes; "
                f"got num_nodes={num_nodes}. Use num_nodes>=6 or set enforce_asymmetry=False."
            )
        if not (0.0 <= edge_prob <= 1.0):
            raise ValueError(f"edge_prob must be in [0, 1], got {edge_prob}")
        if not (0.0 < train_split_ratio < 1.0):
            raise ValueError(
                f"train_split_ratio must be in (0, 1), got {train_split_ratio}"
            )

        self.num_nodes = int(num_nodes)
        self.n_qubits = self.num_nodes * (self.num_nodes - 1) // 2
        self.edge_prob = float(edge_prob)
        self.train_split_ratio = float(train_split_ratio)
        self.seed = int(seed)
        self.enforce_asymmetry = bool(enforce_asymmetry)
        self.active_split = "train"

        self.seed_graph = self._generate_seed_graph()
        self.seed_adjacency = nx.to_numpy_array(
            self.seed_graph, nodelist=range(self.num_nodes), dtype=np.int8
        )

        all_data = self._enumerate_permuted_bitstrings()
        rng = np.random.default_rng(self.seed)
        shuffled = all_data[rng.permutation(len(all_data))]

        split_idx = int(len(shuffled) * self.train_split_ratio)
        self.train_data = shuffled[:split_idx].astype(np.int8)
        self.test_data = shuffled[split_idx:].astype(np.int8)
        self.all_data = shuffled.astype(np.int8)

        self._valid_patterns = set(map(tuple, self.all_data.astype(int)))
        self._train_patterns = set(map(tuple, self.train_data.astype(int)))
        self._test_patterns = set(map(tuple, self.test_data.astype(int)))
        self.data = self.train_data

    def _generate_seed_graph(self) -> nx.Graph:
        """Generate a deterministic Erdős-Rényi seed graph."""
        rng = np.random.default_rng(self.seed)
        max_attempts = 10_000

        for _ in range(max_attempts):
            graph_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            graph = nx.erdos_renyi_graph(
                n=self.num_nodes, p=self.edge_prob, seed=graph_seed
            )
            if not self.enforce_asymmetry or self._is_asymmetric(graph):
                return graph

        raise RuntimeError(
            "Failed to generate an asymmetric seed graph after "
            f"{max_attempts} attempts. Try a different seed or edge_prob."
        )

    @staticmethod
    def _is_asymmetric(graph: nx.Graph) -> bool:
        """Return true when the graph has exactly one automorphism."""
        matcher = isomorphism.GraphMatcher(graph, graph)
        automorphisms = matcher.isomorphisms_iter()
        next(automorphisms, None)
        return next(automorphisms, None) is None

    def _enumerate_permuted_bitstrings(self) -> np.ndarray:
        """Enumerate and deduplicate all permuted upper-triangle bitstrings."""
        if self.num_nodes > 9:
            raise ValueError(
                "GraphIsomorphismDataset enumerates all node permutations; "
                f"num_nodes={self.num_nodes} would require {factorial(self.num_nodes)} permutations."
            )

        tri_upper = np.triu_indices(self.num_nodes, k=1)
        samples = np.empty((factorial(self.num_nodes), self.n_qubits), dtype=np.int8)

        for row_idx, perm in enumerate(permutations(range(self.num_nodes))):
            permuted = self.seed_adjacency[np.ix_(perm, perm)]
            samples[row_idx] = permuted[tri_upper]

        return np.unique(samples, axis=0).astype(np.int8)

    def set_split(self, split: str = "train") -> "GraphIsomorphismDataset":
        """
        Set the active split used by ``__len__`` and ``__getitem__``.

        Args:
            split: One of ``"train"``, ``"test"``, or ``"all"``.
        """
        if split == "train":
            self.data = self.train_data
        elif split == "test":
            self.data = self.test_data
        elif split == "all":
            self.data = self.all_data
        else:
            raise ValueError(f"split must be 'train', 'test', or 'all', got {split!r}")

        self.active_split = split
        return self

    def generate(
        self,
        n_samples: int | None = None,
        seed: int = 0,
        split: str = "train",
    ) -> np.ndarray:
        """
        Return a deterministic split, or sample from that split with replacement.

        Args:
            n_samples: Number of samples to draw. If ``None``, returns the full split.
            seed: Random seed used only when sampling.
            split: One of ``"train"``, ``"test"``, or ``"all"``.
        """
        self.set_split(split)
        source = self.data

        if n_samples is None:
            return source

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(source), size=int(n_samples), replace=True)
        self.data = source[indices].astype(np.int8)
        return self.data

    def _bitstring_to_graph(self, bitstring: np.ndarray) -> nx.Graph:
        """Convert one upper-triangle bitstring to a NetworkX graph."""
        bits = np.asarray(bitstring, dtype=np.int8).reshape(-1)
        if bits.shape[0] != self.n_qubits:
            raise ValueError(
                f"Expected bitstring width {self.n_qubits}, got {bits.shape[0]}"
            )

        adjacency = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int8)
        tri_upper = np.triu_indices(self.num_nodes, k=1)
        adjacency[tri_upper] = bits
        adjacency[(tri_upper[1], tri_upper[0])] = bits
        return nx.from_numpy_array(adjacency)

    def evaluate_generation(self, generated_bitstrings: np.ndarray) -> dict[str, float | int]:
        """
        Evaluate generated bitstrings against train/test graph-isomorphism support.

        Returns rates plus raw counts for valid width, isomorphic samples,
        memorized train samples, and held-out test samples.
        """
        samples = np.asarray(generated_bitstrings)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        total_count = len(samples)
        valid_graph_count = 0
        isomorphic_count = 0
        memorized_count = 0
        novel_generalization_count = 0
        invalid_shape_count = 0

        for sample in samples:
            bits = np.asarray(sample).reshape(-1)
            if bits.shape[0] != self.n_qubits:
                invalid_shape_count += 1
                continue

            rounded_bits = (bits.astype(float) >= 0.5).astype(np.int8)
            valid_graph_count += 1
            generated_graph = self._bitstring_to_graph(rounded_bits)

            if nx.is_isomorphic(generated_graph, self.seed_graph):
                isomorphic_count += 1

            sample_tuple = tuple(rounded_bits.astype(int))
            if sample_tuple in self._train_patterns:
                memorized_count += 1
            if sample_tuple in self._test_patterns:
                novel_generalization_count += 1

        denominator = total_count if total_count else 1
        return {
            "valid_graph_rate": valid_graph_count / denominator,
            "isomorphic_rate": isomorphic_count / denominator,
            "novel_generalization_rate": novel_generalization_count / denominator,
            "memorization_rate": memorized_count / denominator,
            "total_count": total_count,
            "valid_graph_count": valid_graph_count,
            "isomorphic_count": isomorphic_count,
            "novel_generalization_count": novel_generalization_count,
            "memorized_count": memorized_count,
            "invalid_shape_count": invalid_shape_count,
        }

    def visualize(self, sample: np.ndarray, ax=None):
        """Visualize a bitstring as the corresponding graph."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        graph = self._bitstring_to_graph(np.asarray(sample))
        pos = nx.spring_layout(graph, seed=self.seed)
        nx.draw(
            graph,
            pos,
            ax=ax,
            with_labels=True,
            node_color="#7BB7D9",
            node_size=400,
            edge_color="gray",
            font_size=8,
            font_weight="bold",
        )
        ax.set_title(f"Graph Isomorphism (N={self.num_nodes})")
        return ax
