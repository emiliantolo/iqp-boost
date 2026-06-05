import pytest

from src.core.circuits import setup_iqp_circuit


def test_aachen_heavy_hex_builds_single_layer_gates():
    _, gates, desc, wires = setup_iqp_circuit(20, topology="aachen_heavy_hex")

    assert len(gates) == 39
    assert "Aachen heavy-hex topology" in desc
    assert "one_qubit=20, two_qubit=19" in desc
    assert "layers=" not in desc
    assert wires is None


def test_aachen_topology_name_is_not_public_interface():
    with pytest.raises(ValueError, match="Unknown topology: aachen"):
        setup_iqp_circuit(20, topology="aachen")
