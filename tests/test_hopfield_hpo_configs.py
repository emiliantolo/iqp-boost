import json
from pathlib import Path


CONFIGS = [
    ("hopfield_20q_p1_b15", 1, 1.5),
    ("hopfield_20q_p1_b20", 1, 2.0),
    ("hopfield_20q_p2_b15", 2, 1.5),
    ("hopfield_20q_p2_b20", 2, 2.0),
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

LEAN_FIXED = {
    "turbo": 32,
    "monitor_interval": None,
    "compute_snr": False,
    "clear_jax_caches": True,
}

RETRAIN_OVERRIDES = {
    "turbo": 10,
    "monitor_interval": 10,
    "compute_snr": True,
    "clear_jax_caches": True,
}


def test_hopfield_20q_hpo_configs_are_split_by_topology_and_include_importance_sampling():
    root = Path("configs/hpo/hopfield")
    assert {path.name for path in root.glob("hopfield_20q*.json")} == set()

    for topology, expected_circuit_config in TOPOLOGIES.items():
        folder = root / topology
        expected_files = {f"{stem}_{topology}.json" for stem, _, _ in CONFIGS}
        assert {path.name for path in folder.glob("*.json")} == expected_files

        for stem, expected_patterns, expected_beta in CONFIGS:
            config = json.loads((folder / f"{stem}_{topology}.json").read_text())
            fixed = config["fixed_config"]
            search = config["search_space"]
            params = config["dataset"]["params"]

            assert config["study_name"] == f"{stem}_{topology}"
            assert config["objective_metric"] == "tvd_exact"
            assert fixed["baseline"] == "none"
            assert fixed["exact_sampling"] is True
            assert fixed["skip_sampling"] is True
            assert fixed["final_eval_sampling"] is True
            assert fixed["report_fcfw"] is True
            assert fixed["dynamic_is_within_shell"] == "shell_uniform"
            assert fixed["train_samples"] == 1000
            assert fixed["shots"] == 1000
            for key, value in LEAN_FIXED.items():
                assert fixed[key] == value
            assert fixed["circuit_config"] == expected_circuit_config
            assert params["n_qubits"] == 20
            assert params["n_patterns"] == expected_patterns
            assert params["beta"] == expected_beta
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
            assert search["lambda_schedule.gamma"] == {
                "type": "float",
                "low": 0.01,
                "high": 1.0,
                "log": True,
            }
            assert search["lambda_schedule.tau"] == {
                "type": "float",
                "low": 0.01,
                "high": 10.0,
                "log": True,
            }
            assert config["best_retrains"]["n_seeds"] == 5
            assert config["best_retrains"]["baseline"] == "standalone"
            assert config["best_retrains"]["config_overrides"] == RETRAIN_OVERRIDES
