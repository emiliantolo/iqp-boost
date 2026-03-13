"""IQP Primitive Layer: Circuit setup and parameter initialization."""

import numpy as np
import jax
import networkx as nx
import iqpopt as iqp
from iqpopt.utils import nearest_neighbour_gates, local_gates, random_gates, initialize_from_data, expand_gate_list

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
        qubits_in_gate = [q for gen in gate for q in gen if isinstance(q, int)]
        
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


def setup_iqp_circuit(n_qubits: int, topology: str = 'neighbour', n_ancilla: int = 0, **kwargs) -> tuple:
    """Configure IQP circuit gates based on topology.
    
    Args:
        n_qubits: Number of visible qubits (data qubits)
        topology: Gate structure ('neighbour', 'random', 'local')
        n_ancilla: Number of ancilla (hidden) qubits to add. Gates will be
            expanded to couple visible and ancilla qubits.
    
    Returns:
        (circuit, gates, description, wires):
            - circuit: IQP simulator with n_qubits + n_ancilla total qubits
            - gates: Gate structure (expanded if ancillae present)
            - description: Text description of the circuit
            - wires: List of visible qubit indices (None if no ancillae)
    """
    if topology == 'neighbour':
        G = nx.cycle_graph(n_qubits)
        distance = kwargs.get('distance', 1)
        max_weight = kwargs.get('max_weight', 2)
        gates = nearest_neighbour_gates(G, distance=distance, max_weight=max_weight)
        desc = f"Qubits: {n_qubits}\nNeighbour topology: {len(gates)} parameters\n(distance={distance}, max_weight={max_weight})"
    elif topology == 'random':
        n_gates = kwargs.get('n_gates', n_qubits * 2)
        max_idx = kwargs.get('max_idx', n_qubits)
        min_weight = kwargs.get('min_weight', 1)
        max_weight = kwargs.get('max_weight', 2)
        gates = random_gates(n_gates, max_idx=max_idx, min_weight=min_weight, max_weight=max_weight)
        desc = f"Qubits: {n_qubits}\nRandom topology: {len(gates)} parameters\n(n_gates={n_gates},\
            max_idx={max_idx}, weight_range=[{min_weight},{max_weight}])"
    elif topology == 'local':
        max_weight = kwargs.get('max_weight', 2)
        gates = local_gates(n_qubits, max_weight=max_weight)
        desc = f"Qubits: {n_qubits}\nLocal topology: {len(gates)} parameters\n(max_weight={max_weight})"
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Ancilla qubit expansion
    wires = None
    if n_ancilla > 0:
        max_weight_expand = kwargs.get('max_weight', 2)
        gates = expand_gate_list(gates, n_qubits, n_ancilla, max_weight=max_weight_expand)
        wires = list(range(n_qubits))
        total_qubits = n_qubits + n_ancilla
        desc += f"\nAncilla: {n_ancilla} hidden qubits (total={total_qubits}, visible={n_qubits})"
        desc += f"\nExpanded gates: {len(gates)} parameters"
        circuit = iqp.IqpSimulator(total_qubits, gates)
    else:
        circuit = iqp.IqpSimulator(n_qubits, gates)

    return circuit, gates, desc, wires
