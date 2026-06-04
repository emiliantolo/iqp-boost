"""Boosting step lifecycle for IQP ensemble experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import gc

import iqpopt as iqp
import jax
import numpy as np

from src.core import BoostedEnsemble, EvaluationPolicy, compute_lambda_schedule, get_params_init
from src.core.dual_mmd_loss import EnsembleTerms, dual_mmd_loss, gradient_snr
from src.io.reporting import (
    report_acceptance,
    report_gradient_snr,
    report_loss_components,
    report_rejection,
    report_step,
)
from src.core import WeightStrategyContext, WeightStrategyResult, apply_weight_strategy


@dataclass
class BoostingStepContext:
    config: dict
    ensemble: BoostedEnsemble
    x_train: np.ndarray
    key: jax.Array
    evaluation: EvaluationPolicy
    metrics_history: dict
    step: int
    monitor_interval: int | None
    turbo_opt: int | None
    acceptance_metric: str
    min_alpha_accept: float
    previous_stats: dict
    previous_training_mmd: float
    validity_fn: Callable | None = None
    coverage_fn: Callable | None = None
    top_k_tvd_fn: Callable | None = None
    step_prefix: str = "Step"
    include_sampled_tvd: bool = True
    apply_lambda_schedule: bool = True
    compute_snr: bool = True
    run_cleanup: bool = True


@dataclass
class BoostingStepResult:
    key: jax.Array
    alpha: float
    accepted: bool
    should_stop: bool
    training_mmd: float
    stats: dict
    previous_stats: dict
    previous_training_mmd: float


@dataclass
class InitialBoostingResult:
    key: jax.Array
    alpha: float
    training_mmd: float
    stats: dict
    metrics_history: dict


def initialize_boosting_ensemble(
    *,
    ensemble: BoostedEnsemble,
    x_train: np.ndarray,
    key: jax.Array,
    config: dict,
    monitor_interval: int | None,
    turbo_opt: int | None,
    evaluation: EvaluationPolicy,
    top_k_tvd_fn: Callable | None = None,
    print_model0_loss: bool = True,
    include_sampled_tvd: bool = True,
) -> InitialBoostingResult:
    key, trainer, alpha = train_ensemble_model_0(ensemble, x_train, key, config, monitor_interval, turbo_opt)

    losses = getattr(trainer, "losses", [])
    if print_model0_loss and len(losses) > 0:
        print(f"Model 0 final loss: {float(losses[-1]):.6f}")

    training_mmd = evaluation.evaluate_ensemble_training_mmd(ensemble)
    step_model_probs = (
        evaluation.exact_ensemble_probs(ensemble)
        if hasattr(evaluation, "exact_metrics_enabled") and evaluation.exact_metrics_enabled("steps")
        else None
    )
    if not evaluation.sampling_enabled:
        stats = evaluation.analytical_mmd_stats(training_mmd, model_probs=step_model_probs)
    else:
        _, stats = evaluation.sample_and_evaluate_ensemble(ensemble, step=0)

    report_step(
        0,
        config['n_models'],
        training_mmd=training_mmd,
        sampled_mmd=stats['mmd'] if evaluation.sampling_enabled else None,
        sampled_tvd=stats.get('tvd') if evaluation.sampling_enabled and include_sampled_tvd else None,
        alpha_opt=alpha,
        oracle_tvd=top_k_tvd_fn(1) if top_k_tvd_fn else None,
    )

    history = {k: [v] for k, v in stats.items()}
    history['step'] = [0]
    history['alpha'] = [alpha]
    history['training_loss'] = [training_mmd]
    history['accepted_by_metric'] = [True]

    return InitialBoostingResult(
        key=key,
        alpha=alpha,
        training_mmd=training_mmd,
        stats=stats,
        metrics_history=history,
    )


def run_boosting_step(context: BoostingStepContext) -> BoostingStepResult:
    step = context.step
    ensemble = context.ensemble
    snapshot = ensemble.snapshot_state()

    if context.apply_lambda_schedule:
        _apply_lambda_schedule(context)
    key, alpha = train_candidate_model(context)

    if alpha <= context.min_alpha_accept:
        print(
            f"  [{context.step_prefix} {step}] REJECTED "
            f"(alpha_opt={alpha:.3e} <= {context.min_alpha_accept:.1e})"
        )
        ensemble.restore_state(snapshot)
        return BoostingStepResult(
            key=key,
            alpha=alpha,
            accepted=False,
            should_stop=False,
            training_mmd=context.previous_training_mmd,
            stats=context.previous_stats,
            previous_stats=context.previous_stats,
            previous_training_mmd=context.previous_training_mmd,
        )

    training_mmd = context.evaluation.evaluate_ensemble_training_mmd(ensemble)
    step_model_probs = (
        context.evaluation.exact_ensemble_probs(ensemble)
        if hasattr(context.evaluation, "exact_metrics_enabled") and context.evaluation.exact_metrics_enabled("steps")
        else None
    )
    if not context.evaluation.sampling_enabled:
        stats = context.evaluation.analytical_mmd_stats(training_mmd, model_probs=step_model_probs)
    else:
        _, stats = context.evaluation.sample_and_evaluate_ensemble(ensemble, step=step)

    report_step(
        step,
        context.config['n_models'],
        training_mmd=training_mmd,
        sampled_mmd=stats['mmd'] if context.evaluation.sampling_enabled else None,
        sampled_tvd=stats.get('tvd') if context.evaluation.sampling_enabled and context.include_sampled_tvd else None,
        alpha_opt=alpha,
        oracle_tvd=context.top_k_tvd_fn(step + 1) if context.top_k_tvd_fn else None,
    )

    accepted, should_stop, accepted_by_metric = _check_acceptance(
        context=context,
        current_stats=stats,
        current_training_mmd=training_mmd,
        snapshot=snapshot,
    )

    if should_stop:
        return BoostingStepResult(
            key=key,
            alpha=alpha,
            accepted=accepted,
            should_stop=True,
            training_mmd=training_mmd,
            stats=stats,
            previous_stats=context.previous_stats,
            previous_training_mmd=context.previous_training_mmd,
        )

    if not accepted:
        return BoostingStepResult(
            key=key,
            alpha=alpha,
            accepted=False,
            should_stop=False,
            training_mmd=training_mmd,
            stats=stats,
            previous_stats=context.previous_stats,
            previous_training_mmd=context.previous_training_mmd,
        )

    _append_history(context.metrics_history, step, alpha, training_mmd, stats, accepted_by_metric)
    if context.compute_snr:
        key = _report_step_snr(context, key)
    if context.run_cleanup:
        _cleanup_after_step(step)

    return BoostingStepResult(
        key=key,
        alpha=alpha,
        accepted=True,
        should_stop=False,
        training_mmd=training_mmd,
        stats=stats,
        previous_stats=stats,
        previous_training_mmd=training_mmd,
    )


def compute_dual_components_from_traces(traces: dict, n_samples: int, n_data: int) -> dict:
    """Reconstruct dual loss components from a trace pass."""
    data_vals = []
    ensemble_vals = []
    total_vals = []

    for tr_data, tr_ens, tr_iqp, corr_ens, corr_iqp in zip(
        traces['trs_data'],
        traces['trs_ens'],
        traces['trs_iqp'],
        traces['trs_corr_ens'],
        traces['trs_corr_iqp'],
    ):
        tr_data = np.asarray(tr_data)
        tr_ens = np.asarray(tr_ens)
        tr_iqp = np.asarray(tr_iqp)
        corr_ens = np.asarray(corr_ens)
        corr_iqp = np.asarray(corr_iqp)

        term_p_p = np.mean((tr_iqp * tr_iqp - corr_iqp) * n_samples / (n_samples - 1))
        cross_data = np.mean(tr_iqp * tr_data)
        cross_ens = np.mean(tr_iqp * tr_ens)
        term_data_data = np.mean((tr_data * tr_data * n_data - 1) / (n_data - 1))
        term_ens_ens = np.mean((tr_ens * tr_ens - corr_ens) * n_samples / (n_samples - 1))

        data_val = term_p_p - 2.0 * cross_data + term_data_data
        ensemble_val = term_p_p - 2.0 * cross_ens + term_ens_ens

        data_vals.append(data_val)
        ensemble_vals.append(ensemble_val)
        total_vals.append(data_val - ensemble_val)

    return {
        'total': float(np.mean(total_vals)),
        'data': float(np.mean(data_vals)),
        'ensemble': float(np.mean(ensemble_vals)),
    }


def train_ensemble_model_0(
    ensemble: BoostedEnsemble,
    x_train: np.ndarray,
    key: jax.Array,
    config: dict,
    monitor_interval: int | None,
    turbo_opt: int | None,
) -> tuple:
    """Train the first model of the ensemble."""
    key, init_key = jax.random.split(key, 2)
    params_init = get_params_init(config.get('init_baseline', 'random'), ensemble.iqp_circuit, x_train, init_key)

    caching_level = config.get('caching_level', 'none')
    stochastic_ops = caching_level == 'none'

    key, ops_key = jax.random.split(key)
    ensemble.terms.sample_ops(ensemble.iqp_circuit, ensemble.sigma, ensemble.n_ops, ops_key, wires=ensemble.wires)

    loss_kwargs = {
        "params": params_init,
        "iqp_circuit": ensemble.iqp_circuit,
        "weights": [],
        "ground_truth": x_train,
        "ensemble_terms": ensemble.terms,
        "sigma": ensemble.sigma,
        "n_ops": ensemble.n_ops,
        "n_samples": ensemble.n_samples,
        "lambda_dual": ensemble.lambda_dual,
        "key": key,
        "stochastic_ops": stochastic_ops,
        "ensemble_models": [],
        "wires": ensemble.wires,
        "max_batch_ops": config.get('max_batch_ops', None),
    }

    trainer = iqp.Trainer("Adam", dual_mmd_loss, stepsize=config['learning_rate'])
    trainer.train(
        n_iters=config.get('epochs_per_step', 250),
        loss_kwargs=loss_kwargs,
        monitor_interval=monitor_interval,
        turbo=turbo_opt,
    )

    key, subkey = jax.random.split(key, 2)
    gamma, tau = _alpha_schedule_params(config)
    alpha = ensemble.add_model(trainer.final_params, subkey, gamma=gamma, tau=tau)

    ensemble.training_losses.append({
        'total': np.array(trainer.losses),
        'data_final': None,
        'ensemble_final': None,
    })

    return key, trainer, alpha


def train_candidate_model(context: BoostingStepContext) -> tuple[jax.Array, float]:
    """Train and add one candidate model, then apply the configured weight strategy."""
    ensemble = context.ensemble
    config = context.config
    key, step_key = jax.random.split(context.key)
    step_key, init_key = jax.random.split(step_key)

    params_init = get_params_init(config['init_later'], ensemble.iqp_circuit, context.x_train, init_key)

    caching_level = config.get('caching_level', 'none')
    if caching_level == 'step' and ensemble.models:
        ensemble.refresh_terms(step_key)
    stochastic_ops = caching_level == 'none'

    loss_kwargs = {
        "params": params_init,
        "iqp_circuit": ensemble.iqp_circuit,
        "weights": ensemble.weights,
        "ground_truth": context.x_train,
        "ensemble_terms": ensemble.terms,
        "sigma": ensemble.sigma,
        "n_ops": ensemble.n_ops,
        "n_samples": ensemble.n_samples,
        "lambda_dual": ensemble.lambda_dual,
        "key": step_key,
        "stochastic_ops": stochastic_ops,
        "ensemble_models": ensemble.models,
        "wires": ensemble.wires,
        "max_batch_ops": config.get('max_batch_ops', None),
        "max_batch_samples": config.get('max_batch_samples', None),
    }

    monitor_interval = config.get('monitor_interval', context.turbo_opt)
    trainer = iqp.Trainer("Adam", dual_mmd_loss, stepsize=config['learning_rate'])
    trainer.train(
        n_iters=config['epochs_per_step'],
        loss_kwargs=loss_kwargs,
        monitor_interval=monitor_interval,
        turbo=context.turbo_opt,
    )

    data_hist, ens_hist, hist_epochs = _compute_component_history(
        context=context,
        trainer=trainer,
        step_key=step_key,
        stochastic_ops=stochastic_ops,
    )
    traces = dual_mmd_loss(
        trainer.final_params,
        ensemble.iqp_circuit,
        context.x_train,
        ensemble.terms,
        ensemble.weights,
        ensemble.sigma,
        ensemble.n_ops,
        ensemble.n_samples,
        step_key,
        lambda_dual=ensemble.lambda_dual,
        return_traces=True,
        wires=ensemble.wires,
        stochastic_ops=stochastic_ops,
        ensemble_models=ensemble.models,
        max_batch_ops=ensemble.max_batch_ops,
        max_batch_samples=ensemble.max_batch_samples,
    )

    final_comp = compute_dual_components_from_traces(
        traces,
        n_samples=ensemble.n_samples,
        n_data=len(context.x_train),
    )

    samples_old, samples_new = _sample_for_weight_search(context, trainer.final_params)

    key, subkey = jax.random.split(key, 2)
    gamma, tau = _alpha_schedule_params(config)
    alpha = ensemble.add_model(trainer.final_params, subkey, gamma=gamma, tau=tau)
    weight_result = _apply_weight_strategy(context, traces, samples_old, samples_new)
    alpha = weight_result.alpha

    ensemble.training_losses.append({
        'total': np.array(trainer.losses),
        'data_hist': np.array(data_hist) if data_hist else None,
        'ens_hist': np.array(ens_hist) if ens_hist else None,
        'hist_epochs': np.array(hist_epochs) if hist_epochs else None,
        'data_final': float(final_comp['data']),
        'ensemble_final': float(final_comp['ensemble']),
    })

    if config.get('verbose', False):
        report_loss_components(final_comp['data'], final_comp['ensemble'], trainer.losses[-1])

    return key, alpha


def _apply_lambda_schedule(context: BoostingStepContext) -> None:
    effective_step = len(context.ensemble.models)
    current_lambda = compute_lambda_schedule(
        effective_step,
        context.config['n_models'],
        base_lambda=context.config.get('lambda_dual', 1.0),
        schedule=context.config.get('lambda_schedule', None),
    )
    if current_lambda != context.ensemble.lambda_dual:
        context.ensemble.lambda_dual = current_lambda
        print(f"  lambda_dual: {current_lambda:.4f}")


def _check_acceptance(
    *,
    context: BoostingStepContext,
    current_stats: dict,
    current_training_mmd: float,
    snapshot: dict,
) -> tuple[bool, bool, bool]:
    if context.acceptance_metric == 'training_mmd':
        current_metric = current_training_mmd
        previous_metric = context.previous_training_mmd
    else:
        current_metric = current_stats['mmd']
        previous_metric = context.previous_stats['mmd']

    delta_mmd = previous_metric - current_metric
    accepted = delta_mmd > 0

    if context.config.get('keep_models_for_diagnosis', False):
        report_acceptance(True, delta_mmd, diagnostic_mode=True)
        return True, False, accepted

    if not accepted:
        report_rejection(context.step)
        context.ensemble.restore_state(snapshot)
        return False, context.config.get('stop_on_reject', False), False

    return True, False, True


def _append_history(
    history: dict,
    step: int,
    alpha: float,
    training_mmd: float,
    stats: dict,
    accepted_by_metric: bool,
) -> None:
    for key, value in stats.items():
        if key not in history:
            history[key] = [float('nan')] * len(history['step'])
        history[key].append(value)
    history['step'].append(step)
    history['alpha'].append(alpha)
    history['training_loss'].append(training_mmd)
    history.setdefault('accepted_by_metric', [True] * (len(history['step']) - 1))
    history['accepted_by_metric'].append(bool(accepted_by_metric))


def _compute_component_history(
    *,
    context: BoostingStepContext,
    trainer,
    step_key: jax.Array,
    stochastic_ops: bool,
) -> tuple[list[float], list[float], list[int]]:
    data_hist = []
    ens_hist = []
    hist_epochs = []
    params_hist = getattr(trainer, 'params_hist', None)
    if params_hist is None or len(params_hist) == 0:
        return data_hist, ens_hist, hist_epochs

    actual_params = []
    for epoch, params in enumerate(params_hist):
        if not np.all(params == 0):
            actual_params.append(params)
            hist_epochs.append(epoch)

    if not actual_params:
        return data_hist, ens_hist, hist_epochs

    eval_keys = jax.random.split(step_key, len(actual_params))
    ensemble = context.ensemble
    for params, key in zip(actual_params, eval_keys):
        comp = dual_mmd_loss(
            params,
            ensemble.iqp_circuit,
            context.x_train,
            ensemble.terms,
            ensemble.weights,
            ensemble.sigma,
            ensemble.n_ops,
            ensemble.n_samples,
            key,
            lambda_dual=ensemble.lambda_dual,
            return_components=True,
            stochastic_ops=stochastic_ops,
            ensemble_models=ensemble.models,
            wires=ensemble.wires,
            max_batch_ops=ensemble.max_batch_ops,
            max_batch_samples=ensemble.max_batch_samples,
        )
        data_hist.append(float(comp['data']))
        ens_hist.append(float(comp['ensemble']))
    return data_hist, ens_hist, hist_epochs


def _sample_for_weight_search(context: BoostingStepContext, params) -> tuple[np.ndarray | None, np.ndarray | None]:
    weight_strategy = context.config.get('weight_strategy', 'greedy')
    sample_based = {'tvd_line_search', 'validity_line_search', 'coverage_line_search', 'sum_line_search'}
    if weight_strategy not in sample_based or not context.ensemble.models:
        return None, None

    ensemble = context.ensemble
    rng = np.random.default_rng(context.config.get('rng_seed', 0) + context.step * 7919)
    shots = int(context.config.get('shots', 1000))
    samples_old = ensemble.sample(shots, rng)
    try:
        samples_new = ensemble.iqp_circuit.sample(params, shots=shots, wires=ensemble.wires)
    except TypeError:
        samples_new = ensemble.iqp_circuit.sample(params, shots=shots)
    if ensemble.wires is not None and samples_new.shape[1] != len(ensemble.wires):
        samples_new = samples_new[:, ensemble.wires]
    return samples_old, samples_new


def _apply_weight_strategy(
    context: BoostingStepContext,
    traces: dict,
    samples_old: np.ndarray | None,
    samples_new: np.ndarray | None,
) -> WeightStrategyResult:
    strategy = context.config.get('weight_strategy', 'greedy')
    alpha_grid = int(context.config.get('alpha_n_grid', 11))
    objective_weight = float(context.config.get('alpha_objective_weight', 0.5))
    return apply_weight_strategy(WeightStrategyContext(
        ensemble=context.ensemble,
        strategy=strategy,
        ground_truth=context.x_train,
        trs_old=traces.get('trs_ens'),
        trs_new=traces.get('trs_iqp'),
        trs_data=traces.get('trs_data'),
        trs_corr_old=traces.get('trs_corr_ens'),
        trs_corr_new=traces.get('trs_corr_iqp'),
        samples_old=samples_old,
        samples_new=samples_new,
        validity_fn=context.validity_fn,
        coverage_fn=context.coverage_fn,
        alpha_n_grid=alpha_grid,
        validity_weight=objective_weight,
    ))


def _report_step_snr(context: BoostingStepContext, key: jax.Array) -> jax.Array:
    try:
        key, snr_key = jax.random.split(key)
        ensemble = context.ensemble
        new_params = ensemble.models[-1]
        snr_mmd_samples = int(min(0.1 * ensemble.n_samples, 100))
        snr_terms = EnsembleTerms()
        snr_terms.trs = ensemble.terms.trs[:-1]
        snr_terms.corrs = ensemble.terms.corrs[:-1]
        snr_terms.ops = ensemble.terms.ops

        def dual_loss_for_snr(params, iqp_circuit, x_train, key, sigma, n_ops):
            return dual_mmd_loss(
                params,
                iqp_circuit,
                x_train,
                snr_terms,
                ensemble.weights[:-1],
                sigma,
                n_ops,
                snr_mmd_samples,
                key,
                lambda_dual=ensemble.lambda_dual,
                stochastic_ops=False,
                wires=ensemble.wires,
                max_batch_ops=ensemble.max_batch_ops,
                max_batch_samples=ensemble.max_batch_samples,
            )

        snr_info = gradient_snr(
            new_params,
            ensemble.iqp_circuit,
            context.x_train,
            dual_loss_for_snr,
            key=snr_key,
            sigma=ensemble.sigma,
            n_ops=ensemble.n_ops,
        )
        report_gradient_snr(snr_info)
    except Exception as exc:
        print(f"  [Debug] Gradient SNR computation failed: {str(exc)[:80]}")
    return key


def _cleanup_after_step(step: int) -> None:
    gc.collect()
    if step % 2 == 0:
        jax.clear_caches()


def _alpha_schedule_params(config: dict) -> tuple[float, float]:
    schedule = config.get('lambda_schedule') or {}
    return float(schedule.get('gamma', 2.0)), float(schedule.get('tau', 2.0))
