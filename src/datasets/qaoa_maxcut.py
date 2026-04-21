"""Shallow QAOA MaxCut dataset.

Generates the output probability distribution of a shallow QAOA circuit
applied to the MaxCut problem on a random graph.  The QAOA parameters
(beta, gamma) are set to fixed random values — *not* optimised — to produce
a structured multi-modal interference pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml

from .base import BinaryDataset


class QAOAMaxCutDataset(BinaryDataset):
    """Output distribution of a shallow QAOA circuit for MaxCut.

    A random graph is generated with ``networkx``, the MaxCut cost and mixer
    Hamiltonians are obtained via ``qml.qaoa.maxcut``, and a QAOA circuit of
    depth ``p_depth`` is simulated with fixed random parameters.

    Attributes:
        probs (np.ndarray): Exact probability vector of shape
            ``(2**n_qubits,)`` over the computational basis.
        graph (nx.Graph): The graph used for MaxCut.
    """

    def __init__(self, n_qubits: int = 12,
                 graph_type: str = 'random_regular',
                 graph_seed: int = 42,
                 p_depth: int = 1,
                 param_seed: int = 42):
        """
        Args:
            n_qubits: Number of graph vertices / qubits.  Must be ≤ 20.
            graph_type: ``'random_regular'`` (degree 3) or ``'erdos_renyi'``
                (edge probability 0.5).
            graph_seed: Seed for graph generation.
            p_depth: QAOA circuit depth (number of cost+mixer layer pairs).
            param_seed: Seed for the fixed random QAOA parameters.
        """
        super().__init__()
        if n_qubits > 20:
            raise ValueError(f"n_qubits must be ≤ 20, got {n_qubits}")
        self.n_qubits = n_qubits
        self.graph_type = graph_type
        self.graph_seed = graph_seed
        self.p_depth = p_depth
        self.param_seed = param_seed

        self._valid_patterns = None  # Full-support distribution

        # Generate graph
        self.graph = self._make_graph()

        # Simulate QAOA circuit (cached)
        self.probs = self._load_or_compute()

    def _cache_key(self) -> str:
        import hashlib
        params = (self.n_qubits, self.graph_type, self.graph_seed,
                  self.p_depth, self.param_seed)
        h = hashlib.sha256(repr(params).encode()).hexdigest()[:16]
        return f"qaoa_N{self.n_qubits}_p{self.p_depth}_{h}"

    def _load_or_compute(self) -> np.ndarray:
        """Load cached probs or simulate and cache."""
        from pathlib import Path
        cache_dir = Path("./data/cache/qaoa_maxcut")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self._cache_key()}.npz"

        if cache_file.exists():
            data = np.load(cache_file)
            print(f"[QAOAMaxCutDataset] Loaded from cache: {cache_file}")
            return data['probs']

        probs = self._simulate_qaoa()
        np.savez(cache_file, probs=probs)
        print(f"[QAOAMaxCutDataset] Saved to cache: {cache_file}")
        return probs

    def _make_graph(self) -> nx.Graph:
        """Generate a random graph with networkx."""
        if self.graph_type == 'random_regular':
            # degree 3 random regular graph
            d = 3
            if self.n_qubits <= d:
                d = max(1, self.n_qubits - 1)
                if (self.n_qubits * d) % 2 != 0:
                    d = max(1, d - 1)
            return nx.random_regular_graph(d=d, n=self.n_qubits,
                                           seed=self.graph_seed)
        elif self.graph_type == 'erdos_renyi':
            return nx.erdos_renyi_graph(n=self.n_qubits, p=0.5,
                                        seed=self.graph_seed)
        else:
            raise ValueError(f"Unknown graph_type: {self.graph_type}")

    def _simulate_qaoa(self) -> np.ndarray:
        """Build and simulate the QAOA circuit, return exact output probabilities."""
        cost_h, mixer_h = qml.qaoa.maxcut(self.graph)

        # Fixed random parameters (not optimised)
        rng = np.random.default_rng(self.param_seed)
        gammas = rng.uniform(0, 2 * np.pi, size=self.p_depth)
        betas = rng.uniform(0, np.pi, size=self.p_depth)

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev)
        def qaoa_circuit():
            # Initial |+>^N superposition
            for q in range(self.n_qubits):
                qml.Hadamard(wires=q)

            # QAOA layers
            for layer in range(self.p_depth):
                qml.qaoa.cost_layer(gammas[layer], cost_h)
                qml.qaoa.mixer_layer(betas[layer], mixer_h)

            return qml.probs(wires=range(self.n_qubits))

        probs = np.asarray(qaoa_circuit(), dtype=np.float64)
        return probs

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """Sample bitstrings from the exact QAOA output distribution.

        Args:
            n_samples: Number of samples to draw.
            seed: Random seed for reproducibility.

        Returns:
            Array of shape ``(n_samples, n_qubits)`` with binary values.
        """
        rng = np.random.default_rng(seed)
        p = np.asarray(self.probs, dtype=np.float64)
        p = p / p.sum()
        indices = rng.choice(len(p), size=n_samples, p=p)

        bit_positions = np.arange(self.n_qubits)
        samples = ((indices[:, None] >> bit_positions) & 1).astype(np.int8)

        self.data = samples
        return self.data

    def validity_rate(self, samples: np.ndarray) -> float:
        """Full-support distribution -> always 1.0."""
        return 1.0

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        """Not meaningful for full-support distributions -> always 1.0."""
        return 1.0

    def visualize(self, sample: np.ndarray, ax=None):
        """Visualize a MaxCut partition on the graph."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        partition = sample.astype(int)
        colors = ['#F44336' if partition[i] else '#2196F3'
                  for i in range(self.n_qubits)]

        pos = nx.spring_layout(self.graph, seed=self.graph_seed)
        nx.draw(self.graph, pos, ax=ax, node_color=colors,
                with_labels=True, node_size=400,
                edge_color='gray', font_size=8, font_weight='bold')

        cut_edges = [(u, v) for u, v in self.graph.edges()
                     if partition[u] != partition[v]]
        ax.set_title(f'QAOA MaxCut (N={self.n_qubits}, p={self.p_depth}, '
                     f'cut={len(cut_edges)})')
        return ax
