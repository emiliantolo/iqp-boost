"""Isomorphism-unique random bipartite graph dataset."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms.bipartite.generators import random_graph

from .base import BinaryDataset


class BipartiteGraphDataset(BinaryDataset):
    """
    Generate isomorphism-unique random bipartite graphs across all bipartitions.

    Each graph is encoded as the flattened upper triangle of the full
    ``nodes x nodes`` adjacency matrix.
    """

    def __init__(
        self,
        nodes: int = 8,
        edge_prob: float | tuple[float, float] = (0.25, 0.75),
        n_graphs: int = 160,
        train_split_ratio: float = 0.8,
        seed: int = 42,
        store_graphs: bool = False,
        small_partition_threshold: int = 50,
        max_attempts_per_target: int = 1000,
    ):
        super().__init__()
        if nodes < 2:
            raise ValueError(f"nodes must be at least 2, got {nodes}")
        if n_graphs <= 0:
            raise ValueError(f"n_graphs must be positive, got {n_graphs}")
        if not (0.0 < train_split_ratio < 1.0):
            raise ValueError(
                f"train_split_ratio must be in (0, 1), got {train_split_ratio}"
            )
        if small_partition_threshold < 0:
            raise ValueError(
                f"small_partition_threshold must be non-negative, got {small_partition_threshold}"
            )
        if max_attempts_per_target <= 0:
            raise ValueError(
                f"max_attempts_per_target must be positive, got {max_attempts_per_target}"
            )

        self.nodes = int(nodes)
        self.n_qubits = self.nodes * (self.nodes - 1) // 2
        self.p_min, self.p_max = self._parse_edge_prob(edge_prob)
        self.n_graphs = int(n_graphs)
        self.train_split_ratio = float(train_split_ratio)
        self.seed = int(seed)
        self.store_graphs = bool(store_graphs)
        self.small_partition_threshold = int(small_partition_threshold)
        self.max_attempts_per_target = int(max_attempts_per_target)
        self.active_split = "train"

        self._certificate_to_graphs: dict[tuple, list[nx.Graph]] = defaultdict(list)
        self._accepted_graphs: list[nx.Graph] = []
        self.graphs: list[nx.Graph] = []

        self._build_support()

    @staticmethod
    def _parse_edge_prob(edge_prob: float | tuple[float, float]) -> tuple[float, float]:
        if isinstance(edge_prob, (float, int)):
            p_min = p_max = float(edge_prob)
        else:
            if len(edge_prob) != 2:
                raise ValueError("edge_prob tuple must contain exactly two values")
            p_min, p_max = sorted((float(edge_prob[0]), float(edge_prob[1])))

        if not (0.0 <= p_min <= 1.0 and 0.0 <= p_max <= 1.0):
            raise ValueError("edge_prob values must be in [0, 1]")
        return p_min, p_max

    def _get_all_partitions(self) -> list[tuple[int, int]]:
        return [(n1, self.nodes - n1) for n1 in range(1, self.nodes // 2 + 1)]

    @staticmethod
    def _estimate_partition_capacity(n1: int, n2: int) -> int:
        if n1 == 1:
            return n2 + 1
        edge_slots = n1 * n2
        return min(edge_slots * edge_slots, 5000)

    def _compute_target_distribution(
        self, partitions: list[tuple[int, int]], total_samples: int
    ) -> dict[tuple[int, int], int]:
        caps = np.asarray(
            [self._estimate_partition_capacity(n1, n2) for n1, n2 in partitions],
            dtype=int,
        )
        small_mask = caps <= self.small_partition_threshold
        allocation = np.zeros_like(caps)

        allocation[small_mask] = caps[small_mask]
        remaining = max(0, int(total_samples) - int(allocation.sum()))

        large_indices = np.where(~small_mask)[0]
        if remaining > 0 and len(large_indices) > 0:
            large_caps = caps[large_indices]
            raw = large_caps * remaining / large_caps.sum()
            large_allocation = np.floor(raw).astype(int)
            large_allocation = np.minimum(large_allocation, large_caps)

            remainder = remaining - int(large_allocation.sum())
            if remainder > 0:
                fractional_order = np.argsort(raw - np.floor(raw))[::-1]
                for idx in fractional_order:
                    if remainder <= 0:
                        break
                    if large_allocation[idx] < large_caps[idx]:
                        large_allocation[idx] += 1
                        remainder -= 1

            allocation[large_indices] = large_allocation

        allocation = np.where((caps > 0) & (allocation == 0), 1, allocation)
        return {
            part: int(count)
            for part, count in zip(partitions, allocation)
            if count > 0
        }

    def _sample_edge_prob(self, rng: np.random.Generator) -> float:
        if self.p_min == self.p_max:
            return self.p_min
        return float(rng.uniform(self.p_min, self.p_max))

    def _vectorize(self, graph: nx.Graph) -> np.ndarray:
        matrix = nx.to_numpy_array(graph, nodelist=range(self.nodes), dtype=np.int8)
        return matrix[np.triu_indices(self.nodes, k=1)].astype(np.int8)

    def _vec_to_graph(self, vector: np.ndarray) -> nx.Graph:
        bits = np.asarray(vector, dtype=np.int8).reshape(-1)
        if bits.shape[0] != self.n_qubits:
            raise ValueError(f"Expected vector width {self.n_qubits}, got {bits.shape[0]}")

        matrix = np.zeros((self.nodes, self.nodes), dtype=np.int8)
        rows, cols = np.triu_indices(self.nodes, k=1)
        matrix[rows, cols] = bits
        matrix[cols, rows] = bits
        return nx.from_numpy_array(matrix)

    @staticmethod
    def _component_sizes(graph: nx.Graph) -> tuple[int, ...]:
        return tuple(sorted((len(component) for component in nx.connected_components(graph))))

    def _certificate(self, graph: nx.Graph) -> tuple:
        return (
            graph.number_of_nodes(),
            graph.number_of_edges(),
            tuple(sorted(dict(graph.degree()).values())),
            self._component_sizes(graph),
        )

    def _is_isomorphic_to_any(self, graph: nx.Graph) -> bool:
        cert = self._certificate(graph)
        return any(
            nx.is_isomorphic(graph, existing)
            for existing in self._certificate_to_graphs.get(cert, [])
        )

    def _add_graph_if_unique(self, graph: nx.Graph) -> bool:
        if self._is_isomorphic_to_any(graph):
            return False

        graph_copy = graph.copy()
        cert = self._certificate(graph_copy)
        self._certificate_to_graphs[cert].append(graph_copy)
        self._accepted_graphs.append(graph_copy)
        if self.store_graphs:
            self.graphs.append(graph_copy.copy())
        return True

    def _build_support(self) -> None:
        rng = np.random.default_rng(self.seed)
        partitions = self._get_all_partitions()
        targets = self._compute_target_distribution(partitions, self.n_graphs)

        def try_add_candidate(n1: int, n2: int) -> bool:
            p = self._sample_edge_prob(rng)
            graph_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            graph = random_graph(n1, n2, p, seed=graph_seed)
            graph.add_nodes_from(range(self.nodes))
            return self._is_density_valid(graph) and self._add_graph_if_unique(graph)

        for n1, n2 in partitions:
            target = targets.get((n1, n2), 0)
            added = 0
            attempts = 0
            max_attempts = max(1, target) * self.max_attempts_per_target

            while added < target and attempts < max_attempts and len(self._accepted_graphs) < self.n_graphs:
                if try_add_candidate(n1, n2):
                    added += 1
                attempts += 1

        top_up_attempts = 0
        max_top_up_attempts = self.n_graphs * self.max_attempts_per_target
        while len(self._accepted_graphs) < self.n_graphs and top_up_attempts < max_top_up_attempts:
            n1, n2 = partitions[top_up_attempts % len(partitions)]
            try_add_candidate(n1, n2)
            top_up_attempts += 1

        if len(self._accepted_graphs) < self.n_graphs:
            raise RuntimeError(
                "Failed to generate requested isomorphism-unique bipartite support: "
                f"requested {self.n_graphs}, generated {len(self._accepted_graphs)}. "
                "Try fewer n_graphs, more nodes, a wider edge_prob range, or more attempts."
            )

        vectors = np.asarray(
            [self._vectorize(graph) for graph in self._accepted_graphs[: self.n_graphs]],
            dtype=np.int8,
        )
        graphs = self._accepted_graphs[: self.n_graphs]

        permutation = rng.permutation(len(vectors))
        self.all_data = vectors[permutation].astype(np.int8)
        self._all_graphs = [graphs[int(i)].copy() for i in permutation]

        split_idx = int(len(self.all_data) * self.train_split_ratio)
        self.train_data = self.all_data[:split_idx].astype(np.int8)
        self.test_data = self.all_data[split_idx:].astype(np.int8)
        self._train_graphs = [graph.copy() for graph in self._all_graphs[:split_idx]]
        self._test_graphs = [graph.copy() for graph in self._all_graphs[split_idx:]]

        self._valid_patterns = set(map(tuple, self.all_data.astype(int)))
        self._train_patterns = set(map(tuple, self.train_data.astype(int)))
        self._test_patterns = set(map(tuple, self.test_data.astype(int)))
        self.data = self.train_data

    def set_split(self, split: str = "train") -> "BipartiteGraphDataset":
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

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(source), size=int(n_samples), replace=True)
        self.data = source[indices].astype(np.int8)
        return self.data

    def _bipartite_density(self, graph: nx.Graph) -> float:
        if not nx.is_bipartite(graph):
            return float("nan")
        if graph.number_of_edges() == 0:
            return 0.0

        coloring = nx.bipartite.color(graph)
        left_size = sum(1 for value in coloring.values() if value == 0)
        right_size = graph.number_of_nodes() - left_size
        max_edges = left_size * right_size
        if max_edges == 0:
            return 0.0
        return graph.number_of_edges() / max_edges

    def _is_density_valid(self, graph: nx.Graph) -> bool:
        density = self._bipartite_density(graph)
        return bool(self.p_min <= density <= self.p_max)

    def _is_isomorphic_to_graphs(self, graph: nx.Graph, graphs: Iterable[nx.Graph]) -> bool:
        cert = self._certificate(graph)
        for existing in graphs:
            if cert == self._certificate(existing) and nx.is_isomorphic(graph, existing):
                return True
        return False

    def validity_rate(self, samples: np.ndarray) -> float:
        samples = np.asarray(samples)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        if len(samples) == 0:
            return 0.0

        valid_count = 0
        for sample in samples:
            bits = np.asarray(sample).reshape(-1)
            if bits.shape[0] != self.n_qubits:
                continue
            rounded_bits = (bits.astype(float) >= 0.5).astype(np.int8)
            graph = self._vec_to_graph(rounded_bits)
            if nx.is_bipartite(graph) and self._is_density_valid(graph):
                valid_count += 1
        return valid_count / len(samples)

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        del ground_truth
        samples = np.asarray(samples)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        if len(samples) == 0 or len(self._all_graphs) == 0:
            return 0.0

        covered_indices = set()
        for sample in samples:
            bits = np.asarray(sample).reshape(-1)
            if bits.shape[0] != self.n_qubits:
                continue
            graph = self._vec_to_graph((bits.astype(float) >= 0.5).astype(np.int8))
            cert = self._certificate(graph)
            for idx, support_graph in enumerate(self._all_graphs):
                if idx in covered_indices:
                    continue
                if cert == self._certificate(support_graph) and nx.is_isomorphic(graph, support_graph):
                    covered_indices.add(idx)
                    break
        return len(covered_indices) / len(self._all_graphs)

    def evaluate_generation(self, generated_bitstrings: np.ndarray) -> dict[str, float | int]:
        samples = np.asarray(generated_bitstrings)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        total_count = len(samples)
        bipartite_count = 0
        density_valid_count = 0
        support_isomorphic_count = 0
        memorized_count = 0
        novel_generalization_count = 0
        invalid_shape_count = 0

        for sample in samples:
            bits = np.asarray(sample).reshape(-1)
            if bits.shape[0] != self.n_qubits:
                invalid_shape_count += 1
                continue

            graph = self._vec_to_graph((bits.astype(float) >= 0.5).astype(np.int8))
            is_bipartite = nx.is_bipartite(graph)
            if is_bipartite:
                bipartite_count += 1
            if is_bipartite and self._is_density_valid(graph):
                density_valid_count += 1

            is_train = self._is_isomorphic_to_graphs(graph, self._train_graphs)
            is_test = self._is_isomorphic_to_graphs(graph, self._test_graphs)
            if is_train or is_test:
                support_isomorphic_count += 1
            if is_train:
                memorized_count += 1
            if is_test:
                novel_generalization_count += 1

        denominator = total_count if total_count else 1
        return {
            "bipartite_rate": bipartite_count / denominator,
            "density_valid_rate": density_valid_count / denominator,
            "support_isomorphic_rate": support_isomorphic_count / denominator,
            "memorization_rate": memorized_count / denominator,
            "novel_generalization_rate": novel_generalization_count / denominator,
            "total_count": total_count,
            "bipartite_count": bipartite_count,
            "density_valid_count": density_valid_count,
            "support_isomorphic_count": support_isomorphic_count,
            "memorized_count": memorized_count,
            "novel_generalization_count": novel_generalization_count,
            "invalid_shape_count": invalid_shape_count,
        }

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
        ax.set_title(f"Bipartite Graph (N={self.nodes})")
        return ax
