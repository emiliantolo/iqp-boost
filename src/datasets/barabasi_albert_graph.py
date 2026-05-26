"""Barabasi-Albert preferential attachment graph dataset."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .base import BinaryDataset


class BarabasiAlbertGraphDataset(BinaryDataset):
    """
    Generate Barabasi-Albert preferential attachment graphs.

    Each graph is encoded as the flattened upper triangle of the full
    ``nodes x nodes`` adjacency matrix.
    """

    def __init__(
        self,
        nodes: int = 8,
        m: int = 2,
        n_graphs: int = 160,
        train_split_ratio: float = 0.8,
        seed: int = 42,
        store_graphs: bool = False,
    ):
        super().__init__()
        if nodes < 2:
            raise ValueError(f"nodes must be at least 2, got {nodes}")
        if m <= 0:
            raise ValueError("m must be positive")
        if m >= nodes:
            raise ValueError("m must be smaller than nodes")
        if n_graphs <= 1:
            raise ValueError(f"n_graphs must be greater than 1, got {n_graphs}")
        if not (0.0 < train_split_ratio < 1.0):
            raise ValueError(f"train_split_ratio must be in (0, 1), got {train_split_ratio}")

        self.nodes = int(nodes)
        self.m = int(m)
        self.n_graphs = int(n_graphs)
        self.train_split_ratio = float(train_split_ratio)
        self.seed = int(seed)
        self.store_graphs = bool(store_graphs)
        self.n_qubits = self.nodes * (self.nodes - 1) // 2
        self.active_split = "train"
        self.graphs: list[nx.Graph] = []

        self._build_support()

    def _generate_graph(self, rng: np.random.Generator) -> nx.Graph:
        graph_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        return nx.barabasi_albert_graph(self.nodes, self.m, seed=graph_seed)

    def _vectorize(self, graph: nx.Graph) -> np.ndarray:
        matrix = nx.to_numpy_array(graph, nodelist=range(self.nodes), dtype=np.int8)
        return matrix[np.triu_indices(self.nodes, k=1)].astype(np.int8)

    def _vec_to_graph(self, vector: np.ndarray) -> nx.Graph:
        bits = np.asarray(vector).reshape(-1)
        if bits.shape[0] != self.n_qubits:
            raise ValueError(f"Expected vector width {self.n_qubits}, got {bits.shape[0]}")

        bits = (bits.astype(float) >= 0.5).astype(np.int8)
        matrix = np.zeros((self.nodes, self.nodes), dtype=np.int8)
        rows, cols = np.triu_indices(self.nodes, k=1)
        matrix[rows, cols] = bits
        matrix[cols, rows] = bits
        return nx.from_numpy_array(matrix)

    def _build_support(self) -> None:
        rng = np.random.default_rng(self.seed)
        graphs = [self._generate_graph(rng) for _ in range(self.n_graphs)]
        vectors = np.asarray([self._vectorize(graph) for graph in graphs], dtype=np.int8)
        permutation = rng.permutation(len(vectors))

        self.all_data = vectors[permutation].astype(np.int8)
        self._all_graphs = [graphs[int(i)].copy() for i in permutation]
        if self.store_graphs:
            self.graphs = [graph.copy() for graph in self._all_graphs]

        split_idx = int(len(self.all_data) * self.train_split_ratio)
        if split_idx <= 0 or split_idx >= len(self.all_data):
            raise ValueError("train_split_ratio produced an empty train or test split")

        self.train_data = self.all_data[:split_idx].astype(np.int8)
        self.test_data = self.all_data[split_idx:].astype(np.int8)
        self._train_graphs = [graph.copy() for graph in self._all_graphs[:split_idx]]
        self._test_graphs = [graph.copy() for graph in self._all_graphs[split_idx:]]

        self._valid_patterns = set(map(tuple, self.all_data.astype(int)))
        self._train_patterns = set(map(tuple, self.train_data.astype(int)))
        self._test_patterns = set(map(tuple, self.test_data.astype(int)))
        self.data = self.train_data

    def set_split(self, split: str = "train") -> "BarabasiAlbertGraphDataset":
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
        self.set_split(split)
        source = self.data
        if n_samples is None:
            return source
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(source), size=int(n_samples), replace=True)
        self.data = source[indices].astype(np.int8)
        return self.data

    @staticmethod
    def _degree_gini(graph: nx.Graph) -> float:
        degrees = np.asarray([degree for _, degree in graph.degree()], dtype=np.float64)
        if len(degrees) == 0 or float(degrees.sum()) == 0.0:
            return 0.0
        diffs = np.abs(degrees[:, None] - degrees[None, :])
        return float(diffs.sum() / (2.0 * len(degrees) * degrees.sum()))

    @staticmethod
    def _mean_abs_stat_error(
        target_graphs: list[nx.Graph],
        model_graphs: list[nx.Graph],
        stat_fn,
    ) -> float:
        if not target_graphs or not model_graphs:
            return 0.0
        return float(abs(np.mean([stat_fn(graph) for graph in target_graphs]) - np.mean([stat_fn(graph) for graph in model_graphs])))

    def _degree_histogram_distance(self, target_graphs: list[nx.Graph], model_graphs: list[nx.Graph]) -> float:
        if not target_graphs or not model_graphs:
            return 0.0

        def histogram(graphs: list[nx.Graph]) -> np.ndarray:
            counts = np.zeros(self.nodes, dtype=np.float64)
            for graph in graphs:
                for _, degree in graph.degree():
                    if 0 <= degree < self.nodes:
                        counts[int(degree)] += 1.0
            total = float(counts.sum())
            return counts / total if total > 0.0 else counts

        return float(0.5 * np.abs(histogram(target_graphs) - histogram(model_graphs)).sum())

    def graph_structural_metrics(self, target_samples: np.ndarray, model_samples: np.ndarray) -> dict[str, float]:
        target_graphs = [self._vec_to_graph(sample) for sample in np.asarray(target_samples)]
        model_graphs = [self._vec_to_graph(sample) for sample in np.asarray(model_samples)]
        return {
            "ba_max_degree_error": self._mean_abs_stat_error(
                target_graphs,
                model_graphs,
                lambda graph: max(dict(graph.degree()).values(), default=0),
            ),
            "ba_degree_gini_error": self._mean_abs_stat_error(
                target_graphs,
                model_graphs,
                self._degree_gini,
            ),
            "ba_degree_histogram_distance": self._degree_histogram_distance(target_graphs, model_graphs),
        }

    def evaluate_generation(self, generated_bitstrings: np.ndarray) -> dict[str, float | int]:
        samples = np.asarray(generated_bitstrings)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        total_count = len(samples)
        valid_graph_count = 0
        memorized_count = 0
        novel_generalization_count = 0
        invalid_shape_count = 0
        valid_samples = []

        for sample in samples:
            bits = np.asarray(sample).reshape(-1)
            if bits.shape[0] != self.n_qubits:
                invalid_shape_count += 1
                continue

            rounded_bits = (bits.astype(float) >= 0.5).astype(np.int8)
            valid_graph_count += 1
            valid_samples.append(rounded_bits)

            sample_tuple = tuple(rounded_bits.astype(int))
            if sample_tuple in self._train_patterns:
                memorized_count += 1
            if sample_tuple in self._test_patterns:
                novel_generalization_count += 1

        denominator = total_count if total_count else 1
        metrics = {
            "valid_graph_rate": valid_graph_count / denominator,
            "novel_generalization_rate": novel_generalization_count / denominator,
            "memorization_rate": memorized_count / denominator,
            "total_count": total_count,
            "valid_graph_count": valid_graph_count,
            "novel_generalization_count": novel_generalization_count,
            "memorized_count": memorized_count,
            "invalid_shape_count": invalid_shape_count,
        }
        if valid_samples:
            metrics.update(self.graph_structural_metrics(self.test_data, np.asarray(valid_samples, dtype=np.int8)))
        else:
            metrics.update({
                "ba_max_degree_error": 0.0,
                "ba_degree_gini_error": 0.0,
                "ba_degree_histogram_distance": 0.0,
            })
        return metrics

    def visualize(self, sample: np.ndarray, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        graph = self._vec_to_graph(np.asarray(sample))
        pos = nx.spring_layout(graph, seed=self.seed)
        nx.draw(
            graph,
            pos,
            ax=ax,
            with_labels=True,
            node_color="#9AC7A7",
            node_size=400,
            edge_color="gray",
            font_size=8,
            font_weight="bold",
        )
        ax.set_title(f"Barabasi-Albert (N={self.nodes}, m={self.m})")
        return ax
