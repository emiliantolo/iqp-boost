"""IQP Primitive Layer: Circuit setup and parameter initialization."""

import numpy as np
import jax
import networkx as nx
import numbers
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
        return base_lambda

    start = float(schedule.get('start', base_lambda))
    end   = float(schedule.get('end',   base_lambda))
    kind  = schedule.get('type', 'linear')

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
    else:
        raise ValueError(f"Unknown lambda schedule type: '{kind}'. "
                         f"Use 'linear', 'exponential'.")

    return float(np.clip(value, 0.0, 1.0))


def setup_iqp_circuit(n_qubits: int, topology: str = 'neighbour', n_ancilla: int = 0, **kwargs) -> tuple:
    """Configure IQP circuit gates based on topology.
    
    Args:
        n_qubits: Number of visible qubits (data qubits)
        topology: Gate structure ('neighbour', 'random', 'local')
                n_ancilla: Number of ancilla (hidden) qubits to add.
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
