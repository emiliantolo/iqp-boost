import json
from pathlib import Path


CONFIGS = [
    ("hamming_balls_20q_k4_p008.json", 4, 0.08),
    ("hamming_balls_20q_k4_p012.json", 4, 0.12),
    ("hamming_balls_20q_k8_p008.json", 8, 0.08),
    ("hamming_balls_20q_k8_p012.json", 8, 0.12),
]


def test_hamming_balls_20q_hpo_configs_are_objective_only_and_direct_sigma():
    root = Path("configs/hpo/hamming_balls")
    for filename, expected_k, expected_p in CONFIGS:
        config = json.loads((root / filename).read_text())
        fixed = config["fixed_config"]
        search = config["search_space"]
        params = config["dataset"]["params"]

        assert "pruner" not in config
        assert "sigma_heuristic" not in fixed
        assert config["objective_metric"] == "exact_tvd"
        assert fixed["baseline"] == "none"
        assert fixed["exact_sampling"] is True
        assert fixed["skip_sampling"] is True
        assert fixed["final_eval_sampling"] is True
        assert fixed["report_fcfw"] is False
        assert fixed["dynamic_is_within_shell"] == "shell_uniform"
        assert fixed["train_samples"] == 1000
        assert fixed["shots"] == 1000
        assert fixed["circuit_config"]["topology"] == "aachen_heavy_hex"
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
