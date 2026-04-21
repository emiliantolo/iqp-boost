"""Config-driven CLI for running multiple IQP boosting experiments."""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import datetime
from pathlib import Path

from src.experiment_factory import build_dataset_bundle
from src.runner import run_boosting_experiment


DEFAULT_RUN_CONFIG = {
    'train_samples': 1000,
    'data_seed': 42,
    'n_samples': 1000,
    'n_models': 8,
    'learning_rate': 0.01,
    'epochs_per_step': 200,
    'caching_level': 'none',
    'ensemble_samples_coeff': 1.0,
    'init_baseline': 'covariance',
    'init_later': 'random',
    'lambda_dual': 1.0,
    'lambda_schedule': None,
    'sigma_factor': [0.5],
    'n_ops': 1000,
    'weight_strategy': 'frank_wolfe',
    'alpha_n_grid': 11,
    'alpha_objective_weight': 0.5,
    'baseline': 'standalone',
    'shots': 1000,
    'final_eval_sampling': False,
    'keep_models_for_diagnosis': False,
    'stop_on_reject': False,
    'rng_seed': 42,
    'circuit_config': {'topology': 'neighbour', 'distance': 2, 'max_weight': 2},
    'turbo': 10,
}


def _load_experiment_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == '.json':
        return json.loads(path.read_text())
    if suffix == '.toml':
        import tomllib
        return tomllib.loads(path.read_text())

    raise ValueError("Unsupported config format. Use .json or .toml")


def _sanitize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r'[^a-zA-Z0-9_.-]+', '_', value)
    value = value.strip('._-')
    return value or 'run'


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_scalar(raw_value: str):
    value = raw_value.strip()
    lower = value.lower()

    if lower == 'true':
        return True
    if lower == 'false':
        return False
    if lower in {'none', 'null'}:
        return None

    try:
        if value.startswith('0') and value not in {'0', '0.0'} and not value.startswith('0.'):
            raise ValueError
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if (value.startswith('[') and value.endswith(']')) or (value.startswith('{') and value.endswith('}')):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    return value


def _set_nested_value(target: dict, dotted_key: str, value) -> None:
    parts = [p for p in dotted_key.split('.') if p]
    if not parts:
        raise ValueError('Override key cannot be empty.')

    current = target
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _parse_set_overrides(set_args: list[str]) -> list[tuple[str, object]]:
    overrides = []
    for item in set_args:
        if '=' not in item:
            raise ValueError(f"Invalid --set value '{item}'. Expected key=value.")
        key, raw_value = item.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set value '{item}'. Key cannot be empty.")
        overrides.append((key, _parse_scalar(raw_value)))
    return overrides


def _apply_overrides(run_cfg: dict, overrides: list[tuple[str, object]]) -> dict:
    if not overrides:
        return run_cfg

    result = copy.deepcopy(run_cfg)
    for key, value in overrides:
        _set_nested_value(result, key, value)
    return result


def _build_run_config(defaults: dict, run_spec: dict) -> dict:
    return _deep_merge(defaults, run_spec.get('config', {}))


def _compute_suite_dir(config_path: Path, root_spec: dict, output_override: str | None) -> Path:
    output_spec = root_spec.get('output', {})
    base_dir = Path(output_override or output_spec.get('base_dir', 'out'))

    suite_name = output_spec.get('suite_name') or config_path.stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return base_dir / f"{_sanitize_name(suite_name)}_{timestamp}"


def _select_runs(all_runs: list[dict], only: set[str] | None) -> list[dict]:
    if not only:
        return all_runs

    selected_indices = set()
    selected_names = set()
    for token in only:
        token_str = str(token).strip()
        if token_str.isdigit():
            idx = int(token_str)
            if idx < 0 or idx >= len(all_runs):
                raise ValueError(
                    f"Run index out of range in --only: {idx}. Valid range is [0, {len(all_runs)-1}]"
                )
            selected_indices.add(idx)
        else:
            selected_names.add(token_str)

    selected = []
    for idx, run in enumerate(all_runs):
        run_name = run.get('name', f'run_{idx:02d}')
        if idx in selected_indices or run_name in selected_names:
            selected.append(run)
    return selected


def run_suite(config_path: Path, only: set[str] | None = None,
              dry_run: bool = False, output_dir: str | None = None,
              set_overrides: list[tuple[str, object]] | None = None,
              list_runs_only: bool = False) -> None:
    root = _load_experiment_file(config_path)
    runs = root.get('runs', [])
    if not runs:
        raise ValueError("Config file must contain a non-empty 'runs' list.")

    if list_runs_only:
        print("Available runs:")
        for idx, run in enumerate(runs):
            run_name = str(run.get('name', f'run_{idx:02d}'))
            dataset_name = str(run.get('dataset', {}).get('name', 'unknown'))
            print(f"- {run_name} (dataset={dataset_name})")
        return

    runs = _select_runs(runs, only)
    if not runs:
        raise ValueError("No runs selected. Check --only names.")

    defaults = _deep_merge(DEFAULT_RUN_CONFIG, root.get('defaults', {}))
    suite_dir = _compute_suite_dir(config_path, root, output_dir)

    print(f"Suite output directory: {suite_dir}")
    if dry_run:
        print("Dry run mode enabled; no experiments will be executed.")

    suite_dir.mkdir(parents=True, exist_ok=True)

    for idx, run_spec in enumerate(runs):
        run_name = _sanitize_name(str(run_spec.get('name', f'run_{idx:02d}')))
        run_cfg = _build_run_config(defaults, run_spec)
        run_cfg = _apply_overrides(run_cfg, set_overrides or [])

        dataset_spec = run_spec.get('dataset', {})
        plot_spec = run_spec.get('plot', {})
        metric_configs = run_spec.get('metric_configs', None)
        baseline_epochs = run_spec.get('baseline_epochs', None)

        bundle = build_dataset_bundle(dataset_spec=dataset_spec, config=run_cfg, plot_spec=plot_spec)

        print(f"\n[{idx + 1}/{len(runs)}] {run_name}: {bundle['dataset_name']}")
        if dry_run:
            continue

        run_boosting_experiment(
            config=run_cfg,
            dataset_name=bundle['dataset_name'],
            x_train=bundle['x_train'],
            validity_fn=bundle['validity_fn'],
            coverage_fn=bundle['coverage_fn'],
            custom_viz_fn=bundle['custom_viz_fn'],
            top_k_tvd_fn=bundle.get('top_k_tvd_fn'),
            metric_configs=metric_configs,
            baseline_epochs=baseline_epochs,
            output_base_dir=str(suite_dir),
            run_name=run_name,
            log_dir=str(suite_dir),
            log_filename='suite.log',
            append_log=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one or more IQP boosting experiments from a JSON/TOML config file."
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to experiment config (.json or .toml).',
    )
    parser.add_argument(
        '--only',
        nargs='*',
        default=None,
        help='Optional list of run names and/or 0-based run indices to execute.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate and print selected runs without executing training.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Override output base directory from config.',
    )
    parser.add_argument(
        '--list-runs',
        action='store_true',
        help='List available run names from the config and exit.',
    )
    parser.add_argument(
        '--set',
        dest='set_values',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help=(
            'Override run config values for all selected runs. '
            'Supports dotted keys (e.g., circuit_config.topology=random, skip_sampling=true). '
            'Repeat --set for multiple overrides.'
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    only = set(args.only) if args.only else None
    set_overrides = _parse_set_overrides(args.set_values)
    run_suite(
        config_path=Path(args.config),
        only=only,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        set_overrides=set_overrides,
        list_runs_only=args.list_runs,
    )


if __name__ == '__main__':
    main()
