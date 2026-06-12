"""Generate 20-qubit tractable dataset experiment configs."""
import json
import os

GRID = [5, 4]
SIGMAS = [0.5, 1.0, 2.0, 4.0]

TOPOLOGIES = {
    "grid2d": {
        "topology": "grid2d",
        "height": 5, "width": 4,
        "distance": 1, "max_weight": 2, "n_ancilla": 0,
    },
    "heavy_hex": {
        "topology": "aachen_heavy_hex",
        "distance": 1, "height": 5, "width": 4,
        "n_ancilla": 0, "num_layers": 1,
    },
}

DATASETS = {
    "hamming_balls": [
        ("k4p008", {"n_qubits": 20, "K": 4, "p": 0.08, "pattern_seed": 0, "radius_fraction": 0.15, "max_exact_states": 2**20}),
        ("k4p012", {"n_qubits": 20, "K": 4, "p": 0.12, "pattern_seed": 0, "radius_fraction": 0.15, "max_exact_states": 2**20}),
        ("k8p008", {"n_qubits": 20, "K": 8, "p": 0.08, "pattern_seed": 0, "radius_fraction": 0.15, "max_exact_states": 2**20}),
        ("k8p012", {"n_qubits": 20, "K": 8, "p": 0.12, "pattern_seed": 0, "radius_fraction": 0.15, "max_exact_states": 2**20}),
    ],
    "hopfield": [
        ("p1b15", {"n_qubits": 20, "n_patterns": 1, "beta": 1.5, "pattern_seed": 42, "max_exact_states": 2**20}),
        ("p2b15", {"n_qubits": 20, "n_patterns": 2, "beta": 1.5, "pattern_seed": 42, "max_exact_states": 2**20}),
        ("p1b20", {"n_qubits": 20, "n_patterns": 1, "beta": 2.0, "pattern_seed": 42, "max_exact_states": 2**20}),
        ("p2b20", {"n_qubits": 20, "n_patterns": 2, "beta": 2.0, "pattern_seed": 42, "max_exact_states": 2**20}),
    ],
    "calorimeter": [
        ("b2th04", {"rows": 5, "cols": 4, "n_blobs": 2, "blob_std_range": [0.3, 1.0], "threshold": 0.4, "momentum_strength": 1.0}),
        ("b2th06", {"rows": 5, "cols": 4, "n_blobs": 2, "blob_std_range": [0.3, 1.0], "threshold": 0.6, "momentum_strength": 1.0}),
    ],
    "ising_spin_glass": [
        ("b10", {"rows": 5, "cols": 4, "beta": 1.0, "coupling_seed": 0}),
        ("b15", {"rows": 5, "cols": 4, "beta": 1.5, "coupling_seed": 0}),
        ("b20", {"rows": 5, "cols": 4, "beta": 2.0, "coupling_seed": 0}),
    ],
    "topological_syndromes": [
        ("p005", {"rows": 5, "cols": 4, "error_rate": 0.05}),
        ("p010", {"rows": 5, "cols": 4, "error_rate": 0.10}),
        ("p015", {"rows": 5, "cols": 4, "error_rate": 0.15}),
    ],
}

EXACT_DATASETS = {"hamming_balls", "hopfield"}

SIGMA_CFG = {
    "type": "spatial_gaussian_mixture",
    "sigma": SIGMAS,
    "lambda": 0,
    "grid_shape": GRID,
    "max_patch_width": 4,
    "max_patch_height": 4,
}

def make_run(name, dset_key, params, topo_key, topo_cfg):
    extra_exact = {}
    if dset_key in EXACT_DATASETS:
        extra_exact = {
            "exact_sampling": True,
            "skip_sampling": False,
            "final_eval_sampling": True,
        }

    return {
        "name": name,
        "dataset": {"name": dset_key, "params": params},
        "plot": {"kind": "none"},
        "config": {
            "train_samples": 1000,
            "data_seed": 42,
            "n_samples": 1000,
            "n_models": 10,
            "learning_rate": 0.01,
            "epochs_per_step": 400,
            "caching_level": "none",
            "init_baseline": "covariance",
            "init_later": "covariance",
            "lambda_dual": 1.0,
            "lambda_schedule": {"type": "frank_wolfe", "gamma": 0.5, "tau": 1.0},
            "sigma": SIGMA_CFG,
            "dynamic_is": True,
            "n_ops": 1024,
            "weight_strategy": "frank_wolfe",
            "baseline": "none",
            "shots": 1000,
            "acceptance_metric": "training_mmd",
            "skip_sampling": False,
            "final_eval_sampling": True,
            "report_fcfw": True,
            "keep_models_for_diagnosis": True,
            "stop_on_reject": False,
            "rng_seed": 42,
            "turbo": 10,
            "circuit_config": topo_cfg,
            "dataset_metrics": True,
            **extra_exact,
        },
    }

configs_dir = os.path.dirname(os.path.abspath(__file__))

for dset_key, param_list in DATASETS.items():
    for param_suffix, params in param_list:
        for topo_key, topo_cfg in TOPOLOGIES.items():
            name = f"{dset_key}_5x4_{param_suffix}_{topo_key}"
            run = make_run(name, dset_key, params, topo_key, topo_cfg)
            doc = {
                "output": {"base_dir": "out_tractable_20q", "suite_name": name},
                "runs": [run],
            }
            filename = os.path.join(configs_dir, f"{name}.json")
            with open(filename, "w") as f:
                json.dump(doc, f, indent=2)
            print(f"  Created {name}.json")

total = 2 * sum(len(v) for v in DATASETS.values())
print(f"\nGenerated {total} configs ({total} experiment runs)")
