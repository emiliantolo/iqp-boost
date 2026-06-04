"""Optuna HPO runner for config-driven IQP boosting experiments."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from src.core import BoostedEnsemble, setup_iqp_circuit
from src.datasets.hopfield import HopfieldDataset
from src.experiments.factory import build_dataset_bundle
from src.experiments.suite import DEFAULT_RUN_CONFIG
from src.datasets.hopfield_evaluation import evaluate_energy_wasserstein, evaluate_memory_recall
from src.run import run_boosting_experiment


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"HPO config not found: {path}")
    if path.suffix.lower() == '.json':
        return json.loads(path.read_text())
    if path.suffix.lower() == '.toml':
        import tomllib
        return tomllib.loads(path.read_text())
    raise ValueError("Unsupported HPO config format. Use .json or .toml")


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_nested(target: dict, dotted_key: str, value: Any) -> None:
    current = target
    parts = dotted_key.split('.')
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _sample_param(trial: optuna.Trial, name: str, spec: dict) -> Any:
    kind = spec.get('type')
    if kind == 'int':
        return trial.suggest_int(name, int(spec['low']), int(spec['high']), step=int(spec.get('step', 1)))
    if kind == 'float':
        return trial.suggest_float(name, float(spec['low']), float(spec['high']), log=bool(spec.get('log', False)))
    if kind == 'categorical':
        return trial.suggest_categorical(name, spec['choices'])
    raise ValueError(f"Unsupported search-space type for {name}: {kind}")


def _trial_config(base_config: dict, search_space: dict, trial: optuna.Trial) -> dict:
    run_config = copy.deepcopy(base_config)
    sampled = {}
    for dotted_key, spec in search_space.items():
        value = _sample_param(trial, dotted_key, spec)
        _set_nested(run_config, dotted_key, value)
        sampled[dotted_key] = value
    trial.set_user_attr('sampled_config', sampled)
    return run_config


def _json_default(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


def _evaluate_best_model_hopfield_metrics(
    hpo_spec: dict,
    best_config_path: Path,
    best_model_path: Path,
    fcfw_weights_list: list | None,
) -> dict | None:
    """Reconstruct the best ensemble and evaluate Hopfield-specific metrics for both normal and FCFW weights."""
    if best_config_path is None or not best_config_path.exists():
        print("[Postprocess] best_config.json missing, skipping Hopfield metrics")
        return None
    if not best_model_path.exists():
        print("[Postprocess] best_model.json missing, skipping Hopfield metrics")
        return None

    dataset_spec = hpo_spec.get('dataset', {})
    if dataset_spec.get('name') != 'hopfield':
        return None

    try:
        best_cfg = json.loads(best_config_path.read_text())
    except Exception as e:
        print(f"[Postprocess] Failed to read best config: {e}")
        return None

    try:
        with open(best_model_path, 'r') as f:
            model_data = json.load(f)
    except Exception as e:
        print(f"[Postprocess] Failed to read best model: {e}")
        return None

    # Rebuild dataset
    ds_params = dict(dataset_spec.get('params', {}))
    test_samples = int(ds_params.pop('test_samples', 0))
    train_samples = int(best_cfg.get('train_samples', 8000))
    if test_samples > 0:
        ds_params['train_split_ratio'] = train_samples / (train_samples + test_samples)
    dataset = HopfieldDataset(**ds_params)

    # Rebuild circuit
    circuit_cfg = best_cfg.get('circuit_config', {})
    n_qubits = dataset.n_qubits
    circuit, _, _, _ = setup_iqp_circuit(n_qubits, **circuit_cfg)

    # Rebuild ensemble
    n_samples = int(best_cfg.get('n_samples', 512))
    ensemble = BoostedEnsemble.load(str(best_model_path), iqp_circuit=circuit, n_samples=n_samples)

    shots = int(best_cfg.get('shots', 10000))
    rng_seed = int(best_cfg.get('rng_seed', 42))
    rng = np.random.default_rng(rng_seed)

    # Normal ensemble
    normal_samples = ensemble.sample(shots, rng)
    normal_metrics = {
        'energy_wasserstein': evaluate_energy_wasserstein(
            dataset, normal_samples, n_baseline=10000, seed=rng_seed + 999
        ),
        **evaluate_memory_recall(dataset, normal_samples),
    }
    normal_metrics.pop('distances_array', None)

    # FCFW ensemble
    fcfw_metrics = {}
    if fcfw_weights_list is not None and len(fcfw_weights_list) > 0:
        fcfw_weights = np.asarray(fcfw_weights_list, dtype=np.float64)
        fcfw_samples = ensemble.sample(shots, rng, weights_override=fcfw_weights)
        fcfw_metrics = {
            'energy_wasserstein': evaluate_energy_wasserstein(
                dataset, fcfw_samples, n_baseline=10000, seed=rng_seed + 1000
            ),
            **evaluate_memory_recall(dataset, fcfw_samples),
        }
        fcfw_metrics.pop('distances_array', None)

    return {
        'ensemble': normal_metrics,
        'ensemble_fcfw': fcfw_metrics,
    }


def _evaluate_best_model_ising_metrics(
    hpo_spec: dict,
    best_config_path: Path,
    best_model_path: Path,
    fcfw_weights_list: list | None,
) -> dict | None:
    """Reconstruct the best ensemble and evaluate Ising-specific metrics for both normal and FCFW weights."""
    if best_config_path is None or not best_config_path.exists():
        print("[Postprocess] best_config.json missing, skipping Ising metrics")
        return None
    if not best_model_path.exists():
        print("[Postprocess] best_model.json missing, skipping Ising metrics")
        return None

    dataset_spec = hpo_spec.get('dataset', {})
    if dataset_spec.get('name') != 'ising':
        return None

    try:
        best_cfg = json.loads(best_config_path.read_text())
    except Exception as e:
        print(f"[Postprocess] Failed to read best config: {e}")
        return None

    try:
        with open(best_model_path, 'r') as f:
            model_data = json.load(f)
    except Exception as e:
        print(f"[Postprocess] Failed to read best model: {e}")
        return None

    from src.datasets.ising import FrustratedIsingDataset
    from src.ising_evaluation import evaluate_pairwise_correlation_error, evaluate_magnetization_absolute_error
    from src.hopfield_evaluation import evaluate_energy_wasserstein

    # Rebuild dataset
    ds_params = dict(dataset_spec.get('params', {}))
    test_samples = int(ds_params.pop('test_samples', 0))
    train_samples = int(best_cfg.get('train_samples', 8000))
    if test_samples > 0:
        ds_params['train_split_ratio'] = train_samples / (train_samples + test_samples)
    dataset = FrustratedIsingDataset(**ds_params)

    # Rebuild circuit
    circuit_cfg = best_cfg.get('circuit_config', {})
    n_qubits = dataset.n_qubits
    circuit, _, _, _ = setup_iqp_circuit(n_qubits, **circuit_cfg)

    # Rebuild ensemble
    n_samples = int(best_cfg.get('n_samples', 512))
    ensemble = BoostedEnsemble.load(str(best_model_path), iqp_circuit=circuit, n_samples=n_samples)

    shots = int(best_cfg.get('shots', 10000))
    rng_seed = int(best_cfg.get('rng_seed', 42))
    rng = np.random.default_rng(rng_seed)

    # Normal ensemble
    normal_samples = ensemble.sample(shots, rng)
    normal_metrics = {
        'energy_wasserstein': evaluate_energy_wasserstein(
            dataset, normal_samples, n_baseline=10000, seed=rng_seed + 999
        ),
        'pairwise_correlation_error': evaluate_pairwise_correlation_error(
            dataset, normal_samples, n_baseline=10000, seed=rng_seed + 999
        ),
        'magnetization_absolute_error': evaluate_magnetization_absolute_error(
            dataset, normal_samples, n_baseline=10000, seed=rng_seed + 999
        ),
    }

    # FCFW ensemble
    fcfw_metrics = {}
    if fcfw_weights_list is not None and len(fcfw_weights_list) > 0:
        fcfw_weights = np.asarray(fcfw_weights_list, dtype=np.float64)
        fcfw_samples = ensemble.sample(shots, rng, weights_override=fcfw_weights)
        fcfw_metrics = {
            'energy_wasserstein': evaluate_energy_wasserstein(
                dataset, fcfw_samples, n_baseline=10000, seed=rng_seed + 1000
            ),
            'pairwise_correlation_error': evaluate_pairwise_correlation_error(
                dataset, fcfw_samples, n_baseline=10000, seed=rng_seed + 1000
            ),
            'magnetization_absolute_error': evaluate_magnetization_absolute_error(
                dataset, fcfw_samples, n_baseline=10000, seed=rng_seed + 1000
            ),
        }

    return {
        'ensemble': normal_metrics,
        'ensemble_fcfw': fcfw_metrics,
    }


def run_hpo(config_path: Path) -> optuna.study.Study:
    hpo_spec = _load_config(config_path)
    study_name = hpo_spec.get('study_name', config_path.stem)
    n_trials = int(hpo_spec.get('n_trials', 60))
    sampler_seed = int(hpo_spec.get('sampler_seed', 42))

    output_base = Path(hpo_spec.get('output_dir', 'out/hpo'))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    hpo_dir = output_base / f"{study_name}_{timestamp}"
    trial_dir = hpo_dir / 'trials'
    hpo_dir.mkdir(parents=True, exist_ok=True)
    trial_dir.mkdir(parents=True, exist_ok=True)

    storage = hpo_spec.get('storage')
    if storage is None:
        storage = f"sqlite:///{hpo_dir / 'study.db'}"

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    base_config = _deep_merge(DEFAULT_RUN_CONFIG, hpo_spec.get('fixed_config', {}))
    dataset_spec = hpo_spec['dataset']
    plot_spec = hpo_spec.get('plot', {})
    search_space = hpo_spec.get('search_space', {})
    objective_metric = hpo_spec.get('objective_metric', 'tvd')

    def objective(trial: optuna.Trial) -> float:
        run_config = _trial_config(base_config, search_space, trial)
        bundle = build_dataset_bundle(dataset_spec=dataset_spec, config=run_config, plot_spec=plot_spec)
        run_name = f"trial_{trial.number:04d}"

        run_kwargs = dict(
            config=run_config,
            dataset=bundle,
            dataset_spec=dataset_spec,
            metric_configs=hpo_spec.get('metric_configs'),
            baseline_epochs=hpo_spec.get('baseline_epochs'),
            output_base_dir=str(trial_dir),
            run_name=run_name,
            log_dir=str(hpo_dir),
            log_filename='hpo.log',
            append_log=True,
        )

        result = run_boosting_experiment(**run_kwargs)

        final_stats = result['final_stats']
        objective_metric = hpo_spec.get('objective_metric', 'tvd')
        metric_value = float(final_stats.get(objective_metric, float('nan')))
        if not math.isfinite(metric_value):
            raise ValueError(f"Trial {trial.number} produced invalid {objective_metric}: {metric_value}")

        output_dir = Path(result['output_dir'])
        model_path = output_dir / 'ensemble.json'
        result['ensemble'].save(str(model_path))

        trial.set_user_attr('final_stats', final_stats)
        trial.set_user_attr('output_dir', str(output_dir))
        trial.set_user_attr('model_path', str(model_path))
        trial.set_user_attr('n_models_accepted', int(result['n_models_accepted']))
        trial.set_user_attr('weights', result['weights'].tolist())
        fcfw_weights = result.get('ensemble_fcfw_weights')
        if fcfw_weights is not None:
            trial.set_user_attr('ensemble_fcfw_weights', np.asarray(fcfw_weights, dtype=np.float64).tolist())
        trial.report(metric_value, step=0)
        return metric_value

    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_model_src = Path(best.user_attrs['model_path'])
    best_model_dst = hpo_dir / 'best_model.json'
    shutil.copyfile(best_model_src, best_model_dst)
    best_config_src = Path(best.user_attrs['output_dir']) / 'config.json'
    best_config_dst = hpo_dir / 'best_config.json'
    if best_config_src.exists():
        shutil.copyfile(best_config_src, best_config_dst)

    best_summary = {
        'study_name': study.study_name,
        'best_trial': best.number,
        f'best_value_{objective_metric}': best.value,
        'best_params': best.params,
        'best_user_attrs': best.user_attrs,
        'best_model_path': str(best_model_dst),
        'best_config_path': str(best_config_dst) if best_config_src.exists() else None,
        'config_path': str(config_path),
    }

    # Evaluate new Hopfield metrics for the best model (normal + FCFW)
    hopfield_metrics = _evaluate_best_model_hopfield_metrics(
        hpo_spec, best_config_dst, best_model_dst, best.user_attrs.get('ensemble_fcfw_weights')
    )
    if hopfield_metrics is not None:
        best_summary['hopfield_metrics'] = hopfield_metrics
        (hpo_dir / 'best_hopfield_metrics.json').write_text(
            json.dumps(hopfield_metrics, indent=2, default=_json_default)
        )
        print("Best model Hopfield metrics saved to best_hopfield_metrics.json")

    # Evaluate new Ising metrics for the best model (normal + FCFW)
    ising_metrics = _evaluate_best_model_ising_metrics(
        hpo_spec, best_config_dst, best_model_dst, best.user_attrs.get('ensemble_fcfw_weights')
    )
    if ising_metrics is not None:
        best_summary['ising_metrics'] = ising_metrics
        (hpo_dir / 'best_ising_metrics.json').write_text(
            json.dumps(ising_metrics, indent=2, default=_json_default)
        )
        print("Best model Ising metrics saved to best_ising_metrics.json")

    (hpo_dir / 'best_trial.json').write_text(json.dumps(best_summary, indent=2, default=_json_default))
    (hpo_dir / 'hpo_config.json').write_text(json.dumps(hpo_spec, indent=2, default=_json_default))
    print(f"Best trial: {best.number} {objective_metric.upper()}={best.value:.6f}")
    print(f"Best model saved to: {best_model_dst}")
    return study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Optuna HPO for IQP boosting.")
    parser.add_argument('--config', required=True, help='Path to HPO config JSON/TOML.')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_hpo(Path(args.config))


if __name__ == '__main__':
    main()
