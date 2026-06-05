import json
from pathlib import Path


CONFIGS = [
    ("hamming_balls_20q_k4_p008", 4, 0.08),
    ("hamming_balls_20q_k4_p012", 4, 0.12),
    ("hamming_balls_20q_k8_p008", 8, 0.08),
    ("hamming_balls_20q_k8_p012", 8, 0.12),
]

TOPOLOGIES = {
    "heavy_hex": {"topology": "aachen_heavy_hex", "n_ancilla": 0},
    "grid_2d": {
        "topology": "grid2d",
        "height": 5,
        "width": 4,
        "distance": 1,
        "max_weight": 2,
        "n_ancilla": 0,
    },
    "local": {"topology": "local", "max_weight": 2, "n_ancilla": 0},
}


def test_hamming_balls_hpo_configs_are_split_by_topology_and_not_duplicated_at_root():
    root = Path("configs/hpo/hamming_balls")
    assert list(root.glob("*.json")) == []

    for topology, expected_circuit_config in TOPOLOGIES.items():
        folder = root / topology
        expected_files = {f"{stem}_{topology}.json" for stem, _, _ in CONFIGS}
        assert {path.name for path in folder.glob("*.json")} == expected_files

        for stem, expected_k, expected_p in CONFIGS:
            config = json.loads((folder / f"{stem}_{topology}.json").read_text())
            fixed = config["fixed_config"]
            search = config["search_space"]
            params = config["dataset"]["params"]

            assert config["study_name"] == f"{stem}_{topology}"
            assert "pruner" not in config
            assert "sigma_heuristic" not in fixed
            assert config["objective_metric"] == "tvd_exact"
            assert fixed["baseline"] == "none"
            assert fixed["exact_sampling"] is True
            assert fixed["skip_sampling"] is True
            assert fixed["final_eval_sampling"] is True
            assert fixed["report_fcfw"] is True
            assert fixed["dynamic_is_within_shell"] == "shell_uniform"
            assert fixed["train_samples"] == 1000
            assert fixed["shots"] == 1000
            assert fixed["circuit_config"] == expected_circuit_config
            assert "num_layers" not in fixed["circuit_config"]
            assert params["n_qubits"] == 20
            assert params["K"] == expected_k
            assert params["p"] == expected_p
            assert search["sigma"]["type"] == "bodyness_sigma"
            assert search["sigma"]["n_sigmas_choices"] == [1, 2, 3]
            assert search["learning_rate"]["log"] is True
            assert search["n_ops"]["choices"] == [512, 1024, 2048]
            assert search["dynamic_is"] == {"type": "categorical", "choices": [False, True]}
            assert search["dynamic_is_beta"] == {
                "type": "float",
                "low": 0.01,
                "high": 0.2,
                "log": True,
            }
            assert config["best_retrains"]["n_seeds"] == 5
            assert config["best_retrains"]["baseline"] == "standalone"
