import json

import pytest

from src.hpo.spec import HpoSpec, load_hpo_spec


def test_load_hpo_spec_resolves_json_defaults(tmp_path):
    config_path = tmp_path / "minimal.json"
    config_path.write_text(json.dumps({"dataset": {"name": "hamming_balls"}}))

    spec = load_hpo_spec(config_path)

    assert isinstance(spec, HpoSpec)
    assert spec.raw == {"dataset": {"name": "hamming_balls"}}
    assert spec.config_path == config_path
    assert spec.study_name == "minimal"
    assert spec.n_trials == 60
    assert spec.sampler_seed == 42
    assert spec.objective.requested_metric == "tvd"
    assert spec.output_base.name == "hpo"
    assert spec.storage is None
    assert spec.direction == "minimize"
    assert spec.dataset_spec == {"name": "hamming_balls"}
    assert spec.plot_spec == {}
    assert spec.fixed_config == {}
    assert spec.search_space == {}
    assert spec.metric_configs is None
    assert spec.baseline_epochs is None


def test_load_hpo_spec_resolves_toml_values(tmp_path):
    config_path = tmp_path / "custom.toml"
    config_path.write_text(
        """
study_name = "custom_hpo"
n_trials = 3
sampler_seed = 9
objective_metric = "test_mmd"
direction = "maximize"
output_dir = "out/custom_hpo"
baseline_epochs = 12

[dataset]
name = "hopfield"

[plot]
kind = "none"

[fixed_config]
n_models = 2

[search_space.learning_rate]
type = "float"
low = 0.001
high = 0.1
log = true
"""
    )

    spec = load_hpo_spec(config_path)

    assert spec.study_name == "custom_hpo"
    assert spec.n_trials == 3
    assert spec.sampler_seed == 9
    assert spec.objective.requested_metric == "test_mmd"
    assert spec.direction == "maximize"
    assert str(spec.output_base) == "out/custom_hpo"
    assert spec.baseline_epochs == 12
    assert spec.dataset_spec == {"name": "hopfield"}
    assert spec.plot_spec == {"kind": "none"}
    assert spec.fixed_config == {"n_models": 2}
    assert spec.search_space["learning_rate"]["type"] == "float"


def test_load_hpo_spec_rejects_unknown_top_level_keys(tmp_path):
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"dataset": {"name": "hamming_balls"}, "pruner": "median"}))

    with pytest.raises(ValueError, match="Unsupported HPO config keys: pruner"):
        load_hpo_spec(config_path)


def test_load_hpo_spec_requires_dataset(tmp_path):
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"study_name": "missing_dataset"}))

    with pytest.raises(ValueError, match="requires dataset"):
        load_hpo_spec(config_path)


@pytest.mark.parametrize("field", ["dataset", "plot", "fixed_config", "search_space", "best_retrains"])
def test_load_hpo_spec_rejects_wrong_dict_field_shape(tmp_path, field):
    config = {"dataset": {"name": "hamming_balls"}, field: []}
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=f"{field!r} must be a dict"):
        load_hpo_spec(config_path)
