import json
from pathlib import Path


GAMMA_SPEC = {"type": "float", "low": 0.01, "high": 1.0, "log": True}
TAU_SPEC = {"type": "float", "low": 0.01, "high": 10.0, "log": True}


def test_frank_wolfe_hpo_configs_use_continuous_schedule_space():
    config_paths = sorted(Path("configs/hpo").rglob("*.json"))
    assert config_paths

    for path in config_paths:
        config = json.loads(path.read_text())
        schedule = config["fixed_config"].get("lambda_schedule")
        if not schedule or schedule.get("type") != "frank_wolfe":
            continue

        search_space = config["search_space"]
        assert search_space["lambda_schedule.gamma"] == GAMMA_SPEC, path
        assert search_space["lambda_schedule.tau"] == TAU_SPEC, path
