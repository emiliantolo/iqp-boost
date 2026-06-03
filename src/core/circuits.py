"""IQP Primitive Layer: Circuit setup and parameter initialization."""

import numpy as np
import jax
import networkx as nx
import numbers
import iqpopt as iqp
from iqpopt.utils import (
    nearest_neighbour_gates,
    local_gates,
    random_gates,
    initialize_from_data,
    expand_gate_list,
    gates_from_covariance,
)
from src.hardware import create_circuit as hardware_create_circuit

def _initialize_with_ancillas(gates, data, n_visible):
    """
    Helper wrapper: filters out ancilla gates, calls the library on visible gates, 
    and reconstructs the parameter array with sensible defaults for ancillas.
    """
    visible_gates = []
    visible_indices = set()
    
    # 1. Filter: Identify gates that only touch visible qubits
    for i, gate in enumerate(gates):
        # Flatten the gate generator to check all qubit indices
        qubits_in_gate = [int(q) for gen in gate for q in gen if isinstance(q, numbers.Integral)]
        
        if all(q < n_visible for q in qubits_in_gate):
            visible_gates.append(gate)
            visible_indices.add(i)

    # 2. Delegate: Use the unmodified library function for the visible subset
    visible_params = initialize_from_data(visible_gates, data) if visible_gates else []
    
    # 3. Reconstruct: Stitch parameters back together in the original order
    final_params = []
    vis_idx = 0
    
    for i, gate in enumerate(gates):
        if i in visible_indices:
            final_params.append(visible_params[vis_idx])
            vis_idx += 1
        else:
            # Simple defaults for ancilla-involved gates
            # Single qubit gates -> pi/4 (mean 0.5), Couplings -> 0.0 (decoupled)
            is_single_qubit_gate = all(len(gen) == 1 for gen in gate)
            final_params.append(np.pi / 4 if is_single_qubit_gate else 0.0)
            
    return np.array(final_params)


def get_params_init(strategy: str, circuit, data, key_rng=None):
    """
    Initialize parameters based on strategy.
    
    Args:
        strategy: 'covariance' or 'random'
        gates: Gate structure from IQP circuit
        data: Training data for covariance-based initialization
        key_rng: JAX random key for 'random' strategy
    
    Returns:
        Parameter array
    """
    if strategy == 'covariance':
        n_visible = data.shape[1]
        if n_visible == circuit.n_qubits:
            return initialize_from_data(circuit.gates, data)
        else:
            return _initialize_with_ancillas(circuit.gates, data, n_visible)
    elif strategy == 'random':
        shape = (len(circuit.gates),)
        if key_rng is not None:
            return jax.random.uniform(key_rng, shape=shape, minval=-np.pi, maxval=np.pi)
        else:
            return np.random.uniform(-np.pi, np.pi, size=shape)
    else:
        raise ValueError(f"Unknown init_strategy: {strategy}")


def aachen_connectivity() -> dict[int, list[int]]:
    """Return the Aachen / heavy-hex style hardware connectivity map."""
    connectivity = {i: [] for i in range(156)}

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
        (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27),
        (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
        (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47),
        (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55),
        (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67),
        (67, 68), (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (73, 74), (74, 75),
        (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87),
        (87, 88), (88, 89), (89, 90), (90, 91), (91, 92), (92, 93), (93, 94), (94, 95),
        (100, 101), (101, 102), (102, 103), (103, 104), (104, 105), (105, 106), (106, 107),
        (107, 108), (108, 109), (109, 110), (110, 111), (111, 112), (112, 113), (113, 114), (114, 115),
        (120, 121), (121, 122), (122, 123), (123, 124), (124, 125), (125, 126), (126, 127),
        (127, 128), (128, 129), (129, 130), (130, 131), (131, 132), (132, 133), (133, 134), (134, 135),
        (140, 141), (141, 142), (142, 143), (143, 144), (144, 145), (145, 146), (146, 147),
        (147, 148), (148, 149), (149, 150), (150, 151), (151, 152), (152, 153), (153, 154), (154, 155),
        (3, 16), (7, 17), (11, 18), (15, 19), (16, 23), (17, 27), (18, 31), (19, 35),
        (21, 36), (25, 37), (29, 38), (33, 39), (36, 41), (37, 45), (38, 49), (39, 53),
        (43, 56), (47, 57), (51, 58), (55, 59), (56, 63), (57, 67), (58, 71), (59, 75),
        (61, 76), (65, 77), (69, 78), (73, 79), (76, 81), (77, 85), (78, 89), (79, 93),
        (83, 96), (87, 97), (91, 98), (95, 99), (96, 103), (97, 107), (98, 111), (99, 115),
        (101, 116), (105, 117), (109, 118), (113, 119), (116, 121), (117, 125), (118, 129), (119, 133),
        (123, 136), (127, 137), (131, 138), (135, 139), (136, 143), (137, 147), (138, 151), (139, 155),
    ]

    for q1, q2 in connections:
        connectivity[q1].append(q2)
        connectivity[q2].append(q1)

    return connectivity


def aachen_graph() -> nx.Graph:
    """Return the full Aachen hardware graph."""
    graph = nx.Graph()
    connectivity = aachen_connectivity()
    graph.add_nodes_from(connectivity)
    for node, neighbours in connectivity.items():
        for neighbour in neighbours:
            graph.add_edge(node, neighbour)
    return graph


def _most_connected_island(graph: nx.Graph, n_qubits: int) -> list[int]:
    """Greedily select a connected subgraph with strong internal connectivity."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if n_qubits > graph.number_of_nodes():
        raise ValueError(
            f"Requested {n_qubits} qubits, but the hardware graph only has {graph.number_of_nodes()} nodes"
        )

    if n_qubits == graph.number_of_nodes():
        return list(graph.nodes())

    start = max(graph.nodes(), key=lambda node: (graph.degree(node), -int(node)))
    selected = [start]
    selected_set = {start}

    while len(selected) < n_qubits:
        frontier = {
            neighbour
            for node in selected
            for neighbour in graph.neighbors(node)
            if neighbour not in selected_set
        }
        if not frontier:
            frontier = set(graph.nodes()) - selected_set

        candidate = max(
            frontier,
            key=lambda node: (
                sum(1 for neighbour in graph.neighbors(node) if neighbour in selected_set),
                graph.degree(node),
                -int(node),
            ),
        )
        selected.append(candidate)
        selected_set.add(candidate)

    return selected


def _physical_qpu_topology(n_qubits: int, distance: int = 1, max_weight: int = 2) -> tuple[list, nx.Graph, list[int]]:
    """Build a topology on the most connected Aachen subgraph."""
    hardware_graph = aachen_graph()
    selected_nodes = _most_connected_island(hardware_graph, n_qubits)
    subgraph = hardware_graph.subgraph(selected_nodes).copy()
    relabel = {node: idx for idx, node in enumerate(selected_nodes)}
    subgraph = nx.relabel_nodes(subgraph, relabel, copy=True)
    gates = nearest_neighbour_gates(subgraph, distance=distance, max_weight=max_weight)
    return gates, subgraph, selected_nodes


def covariance_topology(data, n_gates: int, return_local: bool = True) -> list:
    """Build a data-driven topology from the strongest covariance pairs."""
    return gates_from_covariance(data, n_gates, return_local=return_local)

def compute_lambda_schedule(step: int, n_steps: int, base_lambda: float,
                             schedule: dict | None) -> float:
    """Compute lambda_dual for a given boosting step according to a schedule.

    Args:
        step: Current boosting step index (1-based; model 0 is not scheduled).
        n_steps: Total number of boosting steps (n_models).
        base_lambda: Fallback value (config's lambda_dual) when no schedule is set.
        schedule: Dict with keys:
            type  - 'linear' | 'exponential' (default: 'linear')
            start - lambda at step 1 (default: base_lambda)
            end   - lambda at the final step (default: base_lambda)

    Returns:
        float: lambda_dual interpolated for this step, clamped to [0, 1].
    """
    if not schedule:
        schedule = {}

    start = float(schedule.get('start', base_lambda))
    end   = float(schedule.get('end',   base_lambda))
    kind  = schedule.get('type', 'frank_wolfe')

    # t goes 0 -> 1 across the boosted steps 1 ... n_steps-1
    n_boosted = max(n_steps - 1, 1)
    t = (step - 1) / max(n_boosted - 1, 1)
    t = float(np.clip(t, 0.0, 1.0))

    if kind == 'linear':
        value = start + t * (end - start)
    elif kind == 'exponential':
        if start <= 0 or end <= 0:
            # Fall back to linear if non-positive bounds
            value = start + t * (end - start)
        else:
            value = start * (end / start) ** t
    elif kind == 'frank_wolfe':
        gamma = float(schedule.get('gamma', 2.0))
        tau = float(schedule.get('tau', 2.0))
        alpha = min(1.0, gamma / (step + tau))
        value = 1.0 - alpha
    else:
        raise ValueError(f"Unknown lambda schedule type: '{kind}'. "
                         f"Use 'linear', 'exponential', 'frank_wolfe'.")

    return float(np.clip(value, 0.0, 1.0))



def grid2d_graph(height: int, width: int, periodic: bool = False) -> nx.Graph:
    """Build a 2D rectangular lattice graph.
    
    Args:
        height: Number of rows in the grid
        width: Number of columns in the grid
        periodic: If True, wrap edges (toroidal); False = open grid (default)
    
    Returns:
        NetworkX graph with qubits 0 to height*width-1 mapped as:
        qubit q = i * width + j for position (i, j)
    """
    n_qubits = height * width
    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    
    for i in range(height):
        for j in range(width):
            q = i * width + j
            
            # Right neighbor (within row)
            if j < width - 1:
                q_right = i * width + (j + 1)
                G.add_edge(q, q_right)
            elif periodic:
                q_wrap_right = i * width + 0
                G.add_edge(q, q_wrap_right)
            
            # Bottom neighbor (next row)
            if i < height - 1:
                q_bottom = (i + 1) * width + j
                G.add_edge(q, q_bottom)
            elif periodic:
                q_wrap_bottom = 0 * width + j
                G.add_edge(q, q_wrap_bottom)
    
    return G


def grid2d_topology(height: int, width: int, distance: int = 1, 
                    max_weight: int = 2, periodic: bool = False) -> tuple:
    """Build gate list for 2D grid topology.
    
    Returns (gates, G): gate list and NetworkX graph
    """
    G = grid2d_graph(height, width, periodic=periodic)
    gates = nearest_neighbour_gates(G, distance=distance, max_weight=max_weight)
    return gates, G


def aachen_connectivity() -> dict[int, list[int]]:
    """Return the manually specified IBM Aachen heavy-hex connectivity."""
    connectivity = {i: [] for i in range(156)}

    connections = []
    for row_start in range(0, 141, 20):
        connections.extend((row_start + i, row_start + i + 1) for i in range(15))

    connections.extend([
        (3, 16), (7, 17), (11, 18), (15, 19),
        (16, 23), (17, 27), (18, 31), (19, 35),
        (21, 36), (25, 37), (29, 38), (33, 39),
        (36, 41), (37, 45), (38, 49), (39, 53),
        (43, 56), (47, 57), (51, 58), (55, 59),
        (56, 63), (57, 67), (58, 71), (59, 75),
        (61, 76), (65, 77), (69, 78), (73, 79),
        (76, 81), (77, 85), (78, 89), (79, 93),
        (83, 96), (87, 97), (91, 98), (95, 99),
        (96, 103), (97, 107), (98, 111), (99, 115),
        (101, 116), (105, 117), (109, 118), (113, 119),
        (116, 121), (117, 125), (118, 129), (119, 133),
        (123, 136), (127, 137), (131, 138), (135, 139),
        (136, 143), (137, 147), (138, 151), (139, 155),
    ])

    for q1, q2 in connections:
        connectivity[q1].append(q2)
        connectivity[q2].append(q1)
    return connectivity


def connectivity_gates(connectivity_graph: dict[int, list[int]], num_positions: int,
                       num_layers: int = 1) -> list:
    """Create one-qubit gates plus two-qubit gates for connected pairs."""
    single_qubit_gates = [[[np.int64(i)]] for i in range(num_positions)]
    two_qubit_gates = []
    edges_added = set()

    for q1_key, neighbors in connectivity_graph.items():
        q1 = int(q1_key)
        if not 0 <= q1 < num_positions:
            continue
        for q2_val in neighbors:
            q2 = int(q2_val)
            if not 0 <= q2 < num_positions:
                continue
            edge = tuple(sorted((q1, q2)))
            if edge not in edges_added:
                two_qubit_gates.append([[np.int64(edge[0]), np.int64(edge[1])]])
                edges_added.add(edge)

    two_qubit_gates.sort(key=lambda gate: (gate[0][0], gate[0][1]))
    layer_gates = single_qubit_gates + two_qubit_gates

    final_gates = []
    for _ in range(int(num_layers)):
        final_gates.extend(layer_gates)
    return final_gates


def setup_iqp_circuit(n_qubits: int, topology: str = 'neighbour', n_ancilla: int = 0, **kwargs) -> tuple:
    """Configure IQP circuit gates based on topology.
    
    Args:
        n_qubits: Number of visible qubits (data qubits)
        topology: Gate structure ('neighbour', 'random', 'local', 'grid2d', 'aachen_heavy_hex', 'aachen', 'physical_qpu', 'covariance')
                n_ancilla: Number of ancilla (hidden) qubits to add.
            kwargs.data: Optional training data used by data-driven topologies.
                kwargs.ancilla_topology_mode: How to wire ancilla qubits when n_ancilla > 0:
                        - 'joint' (default): build the selected topology directly on total
                            qubits (visible + ancilla).
                        - 'joint_closed': same as 'joint' but adds a chord edge
                            (V-1, 0) to close the visible sub-ring, preserving
                            the visible-qubit ring inductive bias.
                        - 'expanded': build topology on visible qubits only, then expand
                            gates with ancilla couplings (legacy behavior).
    
    Returns:
        (circuit, gates, description, wires):
            - circuit: IQP simulator with n_qubits + n_ancilla total qubits
            - gates: Gate structure (expanded if ancillae present)
            - description: Text description of the circuit
            - wires: List of visible qubit indices (None if no ancillae)
    """
    ancilla_topology_mode = kwargs.get('ancilla_topology_mode', 'joint')
    if ancilla_topology_mode not in {'expanded', 'joint', 'joint_closed'}:
        raise ValueError("ancilla_topology_mode must be one of: 'joint', 'joint_closed', 'expanded'")

    use_full_topology = n_ancilla > 0 and ancilla_topology_mode in {'joint', 'joint_closed'}
    build_n_qubits = n_qubits + n_ancilla if use_full_topology else n_qubits

    if topology == 'neighbour':
        G = nx.cycle_graph(build_n_qubits)
        distance = kwargs.get('distance', 1)
        max_weight = kwargs.get('max_weight', 2)

        # joint_closed: add chord (V-1, 0) to close the visible sub-ring
        if ancilla_topology_mode == 'joint_closed' and n_ancilla > 0:
            G.add_edge(n_qubits - 1, 0)

        gates = nearest_neighbour_gates(G, distance=distance, max_weight=max_weight)
        desc = f"Qubits: {build_n_qubits}\nNeighbour topology: {len(gates)} parameters\n(distance={distance}, max_weight={max_weight})"
    elif topology == 'random':
        n_gates = kwargs.get('n_gates', build_n_qubits * 2)
        max_idx = kwargs.get('max_idx', build_n_qubits)
        min_weight = kwargs.get('min_weight', 1)
        max_weight = kwargs.get('max_weight', 2)
        gates = random_gates(n_gates, max_idx=max_idx, min_weight=min_weight, max_weight=max_weight)
        desc = f"Qubits: {build_n_qubits}\nRandom topology: {len(gates)} parameters\n(n_gates={n_gates},\
            max_idx={max_idx}, weight_range=[{min_weight},{max_weight}])"
    elif topology == 'local':
        max_weight = kwargs.get('max_weight', 2)
        gates = local_gates(build_n_qubits, max_weight=max_weight)
        desc = f"Qubits: {build_n_qubits}\nLocal topology: {len(gates)} parameters\n(max_weight={max_weight})"
    elif topology == 'grid2d':
        height = kwargs.get('height')
        width = kwargs.get('width')
        
        if height is None or width is None:
            raise ValueError("grid2d requires 'height' and 'width' parameters")
        
        if height * width != n_qubits:
            raise ValueError(
                f"grid2d mismatch: {height}x{width}={height*width} != n_qubits={n_qubits}"
            )
        
        periodic = kwargs.get('periodic', False)
        distance = kwargs.get('distance', 1)
        max_weight = kwargs.get('max_weight', 2)
        
        gates, G = grid2d_topology(height, width, distance=distance, 
                                   max_weight=max_weight, periodic=periodic)
        
        periodic_str = "toroidal" if periodic else "open"
        desc = f"Qubits: {n_qubits} ({height}x{width} {periodic_str} grid)\n"
        desc += f"Grid2D topology: {len(gates)} parameters\n(distance={distance}, max_weight={max_weight})"
    elif topology == 'aachen_heavy_hex':
        if n_ancilla > 0:
            raise ValueError("aachen_heavy_hex currently supports n_ancilla=0 only")
        num_layers = int(kwargs.get('num_layers', 1))
        if n_qubits > 156:
            raise ValueError("aachen_heavy_hex supports at most 156 qubits")
        gates = connectivity_gates(aachen_connectivity(), n_qubits, num_layers=num_layers)
        n_one = n_qubits * num_layers
        n_two = len(gates) - n_one
        desc = (
            f"Qubits: {n_qubits}\n"
            f"Aachen heavy-hex topology: {len(gates)} parameters\n"
            f"layers={num_layers}, one_qubit={n_one}, two_qubit={n_two}"
        )
    elif topology == 'aachen':
        num_layers = kwargs.get('num_layers', 1)
        qpu_circuit = hardware_create_circuit(build_n_qubits, num_layers)
        gates = qpu_circuit.gates
        desc = f"Qubits: {build_n_qubits}\nAachen (hardware.py): {len(gates)} parameters\n(num_layers={num_layers})"
    elif topology == 'physical_qpu':
        distance = kwargs.get('distance', 1)
        max_weight = kwargs.get('max_weight', 2)
        gates, G, selected_nodes = _physical_qpu_topology(build_n_qubits, distance=distance, max_weight=max_weight)
        selected_preview = selected_nodes[:8]
        suffix = '...' if len(selected_nodes) > len(selected_preview) else ''
        desc = (
            f"Qubits: {build_n_qubits} (Aachen/heavy-hex island)\n"
            f"Physical QPU topology: {len(gates)} parameters\n"
            f"(distance={distance}, max_weight={max_weight}, selected={selected_preview}{suffix})"
        )
    elif topology == 'covariance':
        data = kwargs.get('data')
        if data is None:
            raise ValueError("covariance topology requires 'data' in setup_iqp_circuit kwargs")
        if n_ancilla > 0 and ancilla_topology_mode != 'expanded':
            raise ValueError("covariance topology with ancillas requires ancilla_topology_mode='expanded'")
        n_gates = kwargs.get('n_gates', max(n_qubits, 1))
        return_local = kwargs.get('return_local', True)
        gates = covariance_topology(data, n_gates=n_gates, return_local=return_local)
        desc = (
            f"Qubits: {n_qubits}\n"
            f"Covariance topology: {len(gates)} parameters\n"
            f"(n_gates={n_gates}, return_local={return_local})"
        )
    else:
        raise ValueError(f"Unknown topology: {topology}")

    wires = None
    if n_ancilla > 0:
        wires = list(range(n_qubits))
        total_qubits = n_qubits + n_ancilla
        if ancilla_topology_mode == 'expanded':
            max_weight_expand = kwargs.get('max_weight', 2)
            gates = expand_gate_list(gates, n_qubits, n_ancilla, max_weight=max_weight_expand)
            desc += f"\nAncilla: {n_ancilla} hidden qubits (total={total_qubits}, visible={n_qubits})"
            desc += "\nAncilla wiring mode: expanded"
            desc += f"\nExpanded gates: {len(gates)} parameters"
        elif ancilla_topology_mode == 'joint_closed':
            desc += f"\nAncilla: {n_ancilla} hidden qubits (total={total_qubits}, visible={n_qubits})"
            desc += "\nAncilla wiring mode: joint_closed (visible ring preserved)"
        else:
            desc += f"\nAncilla: {n_ancilla} hidden qubits (total={total_qubits}, visible={n_qubits})"
            desc += "\nAncilla wiring mode: joint topology"

        circuit = iqp.IqpSimulator(total_qubits, gates)
    else:
        circuit = iqp.IqpSimulator(n_qubits, gates)

    return circuit, gates, desc, wires
