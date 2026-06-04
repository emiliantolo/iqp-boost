"""Final evaluation and comparison reporting for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core import BoostedEnsemble, EvaluationPolicy
from src.datasets import DatasetBundle
from src.datasets.dataset_metrics import compute_and_save_dataset_metrics
from src.io.reporting import (
    get_plot_config,
    plot_data_ensemble_loss,
    plot_metrics_progression,
    report_final,
    report_metrics_table,
)
from src.core import WeightStrategyContext, apply_weight_strategy


@dataclass(frozen=True)
class FinalEvaluationContext:
    config: dict
    dataset: DatasetBundle
    output: object
    ensemble: BoostedEnsemble
    data_only_ensemble: BoostedEnsemble | None
    data_only_stats: dict | None
    data_only_history: dict | None
    standalone_stats: dict | None
    baseline_train_losses: object
    ensemble_metrics_history: dict
    evaluation: EvaluationPolicy
    sigma: float | list
    shots: int
    rng_seed: int
    reference_stats: dict
    reference_label: str
    metric_configs: list | None = None


@dataclass
class FinalEvaluationResult:
    final_stats: dict
    final_ensemble_samples: np.ndarray | None
    per_model_samples: list
    ensemble_fcfw_stats: dict | None = None
    ensemble_fcfw_weights: np.ndarray | None = None
    data_only_fcfw_stats: dict | None = None
    data_only_fcfw_weights: np.ndarray | None = None


@dataclass(frozen=True)
class FcfwEvaluationResult:
    metrics: dict
    weights: np.ndarray


@dataclass(frozen=True)
class FinalComparison:
    model_rows: list
    table_title: str


def run_final_evaluation(context: FinalEvaluationContext) -> FinalEvaluationResult:
    """Produce final comparison metrics, plots, and CSV output for an experiment run."""
    result = build_final_result(context)
    comparison = build_comparison(context, result)
    publish_final_outputs(context, result, comparison)
    _save_dataset_metrics(context, result)
    return result


def build_final_result(context: FinalEvaluationContext) -> FinalEvaluationResult:
    if context.evaluation.final_sampling_enabled:
        return _sampled_final_result(context)
    return _analytical_final_result(context)


def build_comparison(context: FinalEvaluationContext, result: FinalEvaluationResult) -> FinalComparison:
    model_rows = []
    _add_data_only_rows(context, result, model_rows)
    _add_model_rows(context, model_rows, result.per_model_samples)
    _add_ensemble_fcfw_row(context, result, model_rows)
    return FinalComparison(
        model_rows=model_rows,
        table_title=_table_title(context, result),
    )


def publish_final_outputs(
    context: FinalEvaluationContext,
    result: FinalEvaluationResult,
    comparison: FinalComparison,
) -> None:
    _add_test_metrics(context, result)
    report_metrics_table(context.reference_stats, result.final_stats, comparison.model_rows, comparison.table_title)
    _save_results_csv(context, result)
    _plot_final_progression(context)


def compute_fcfw_stats(
    base_ensemble: BoostedEnsemble,
    x_train: np.ndarray,
    sigma: float | list,
    shots: int | None,
    final_eval_rng: np.random.Generator | None,
    evaluation: EvaluationPolicy,
    sampling_enabled: bool = True,
    evaluation_data: np.ndarray | None = None,
    exact_phase: str = "fcfw",
) -> FcfwEvaluationResult:
    """Compute Fully Corrective Frank-Wolfe weights for an ensemble and evaluate."""
    fcfw_ensemble = BoostedEnsemble(
        base_ensemble.iqp_circuit, base_ensemble.n_models, base_ensemble.sigma, base_ensemble.n_ops,
        base_ensemble.n_samples, base_ensemble.lambda_dual, base_ensemble.wires,
        base_ensemble.max_batch_ops, base_ensemble.max_batch_samples
    )
    fcfw_ensemble.restore_state(base_ensemble.snapshot_state())
    target_data = x_train if evaluation_data is None else evaluation_data
    trs_data = _data_traces(fcfw_ensemble, target_data)

    apply_weight_strategy(WeightStrategyContext(
        ensemble=fcfw_ensemble,
        strategy='fully_corrective',
        trs_data=trs_data,
    ))
    model_probs = (
        evaluation.exact_ensemble_probs(fcfw_ensemble)
        if hasattr(evaluation, "exact_metrics_enabled") and evaluation.exact_metrics_enabled(exact_phase)
        else None
    )
    if sampling_enabled:
        final_fcfw_samples = fcfw_ensemble.sample(shots, final_eval_rng)
        fcfw_stats = evaluation.evaluate_samples(final_fcfw_samples, model_probs=model_probs)
    else:
        fcfw_stats = _analytical_stats(evaluation.evaluate_ensemble_training_mmd(fcfw_ensemble, target_data))
        if model_probs is not None:
            fcfw_stats.update(evaluation.evaluate_exact_probs(model_probs))

    return FcfwEvaluationResult(
        metrics=fcfw_stats,
        weights=np.asarray(fcfw_ensemble.weights, dtype=np.float64),
    )


def _analytical_final_result(context: FinalEvaluationContext) -> FinalEvaluationResult:
    print("\n[skip_sampling=True] Skipping final state vector sampling. Using final training losses.")
    final_loss = (
        context.ensemble_metrics_history['training_loss'][-1]
        if context.ensemble_metrics_history['training_loss']
        else float('nan')
    )
    model_probs = (
        context.evaluation.exact_ensemble_probs(context.ensemble)
        if hasattr(context.evaluation, "exact_metrics_enabled") and context.evaluation.exact_metrics_enabled("final")
        else None
    )
    final_stats = {'mmd': final_loss}
    if model_probs is not None:
        final_stats.update(context.evaluation.evaluate_exact_probs(model_probs))
    report_final(
        context.reference_stats['mmd'],
        final_stats['mmd'],
        len(context.ensemble.models),
        {**final_stats},
    )
    return FinalEvaluationResult(
        final_stats=final_stats,
        final_ensemble_samples=None,
        per_model_samples=[],
    )


def _sampled_final_result(context: FinalEvaluationContext) -> FinalEvaluationResult:
    final_eval_rng = _final_eval_rng(context)
    final_ensemble_samples, _, per_model_samples = context.ensemble.sample(
        context.shots,
        final_eval_rng,
        return_details=True,
    )
    model_probs = (
        context.evaluation.exact_ensemble_probs(context.ensemble)
        if hasattr(context.evaluation, "exact_metrics_enabled") and context.evaluation.exact_metrics_enabled("final")
        else None
    )
    final_stats = context.evaluation.evaluate_samples(final_ensemble_samples, model_probs=model_probs)
    report_final(
        context.reference_stats['mmd'],
        final_stats['mmd'],
        len(context.ensemble.models),
        final_stats,
    )
    return FinalEvaluationResult(
        final_stats=final_stats,
        final_ensemble_samples=final_ensemble_samples,
        per_model_samples=per_model_samples,
    )


def _add_data_only_rows(
    context: FinalEvaluationContext,
    result: FinalEvaluationResult,
    model_rows: list,
) -> None:
    if context.data_only_stats is None or context.reference_label == 'Data-only':
        return

    model_rows.append(("Data-only", context.data_only_stats))
    if not _report_fcfw(context):
        return

    sampled = context.evaluation.final_sampling_enabled
    mode_label = "Sampled" if sampled else "Analytical"
    print(f"\n[FCFW] Computing Fully Corrective Frank-Wolfe weights for Data-only baseline{' (Analytical)' if not sampled else ''}...")
    fcfw_result = compute_fcfw_stats(
        context.data_only_ensemble,
        context.dataset.x_train,
        context.sigma,
        context.shots if sampled else None,
        _final_eval_rng(context) if sampled else None,
        context.evaluation,
        sampling_enabled=sampled,
    )
    result.data_only_fcfw_stats = fcfw_result.metrics
    result.data_only_fcfw_weights = fcfw_result.weights
    _print_fcfw_metrics("Data-only", mode_label, result.data_only_fcfw_stats)
    model_rows.append(("Data-only (FCFW)", result.data_only_fcfw_stats))


def _add_model_rows(
    context: FinalEvaluationContext,
    model_rows: list,
    per_model_samples: list,
) -> None:
    if context.evaluation.final_sampling_enabled:
        for i, model_samples in enumerate(per_model_samples):
            if len(model_samples) > 0:
                model_probs = (
                    context.evaluation.exact_model_probs(
                        context.ensemble.iqp_circuit,
                        context.ensemble.models[i],
                        context.ensemble.wires,
                    )
                    if hasattr(context.evaluation, "exact_metrics_enabled") and context.evaluation.exact_metrics_enabled("per_model")
                    else None
                )
                model_stats = context.evaluation.evaluate_samples(model_samples, model_probs=model_probs)
            else:
                model_stats = {'mmd': float('nan')}
            model_rows.append((f"Model {i}", model_stats))
        return

    for i in range(len(context.ensemble.models)):
        m_loss = (
            context.ensemble_metrics_history['training_loss'][i]
            if i < len(context.ensemble_metrics_history['training_loss'])
            else float('nan')
        )
        model_rows.append((f"Model {i}", {'mmd': m_loss}))


def _add_ensemble_fcfw_row(
    context: FinalEvaluationContext,
    result: FinalEvaluationResult,
    model_rows: list,
) -> None:
    if not _report_fcfw(context):
        return

    sampled = context.evaluation.final_sampling_enabled
    mode_label = "Sampled" if sampled else "Analytical"
    print(f"\n[FCFW] Computing Fully Corrective Frank-Wolfe weights for final ensemble{' (Analytical)' if not sampled else ''}...")
    fcfw_result = compute_fcfw_stats(
        context.ensemble,
        context.dataset.x_train,
        context.sigma,
        context.shots if sampled else None,
        _final_eval_rng(context) if sampled else None,
        context.evaluation,
        sampling_enabled=sampled,
    )
    result.ensemble_fcfw_stats = fcfw_result.metrics
    result.ensemble_fcfw_weights = fcfw_result.weights
    _print_fcfw_metrics("Ensemble", mode_label, result.ensemble_fcfw_stats)
    model_rows.append(("Ensemble (FCFW)", result.ensemble_fcfw_stats))


def _add_test_metrics(context: FinalEvaluationContext, result: FinalEvaluationResult) -> None:
    if context.dataset.x_test is None or len(context.dataset.x_test) == 0:
        return

    test_mmd = context.evaluation.evaluate_ensemble_training_mmd(context.ensemble, context.dataset.x_test)
    result.final_stats['test_mmd'] = test_mmd
    print(f"\n[Test MMD] {test_mmd:.6f}")
    if result.ensemble_fcfw_stats is not None:
        fcfw_test_result = compute_fcfw_stats(
            context.ensemble,
            context.dataset.x_train,
            context.sigma,
            None,
            None,
            context.evaluation,
            sampling_enabled=False,
            evaluation_data=context.dataset.x_test,
        )
        result.final_stats['test_mmd_fcfw'] = fcfw_test_result.metrics["mmd"]
        print(f"[Test MMD FCFW] {result.final_stats['test_mmd_fcfw']:.6f}")


def _save_results_csv(context: FinalEvaluationContext, result: FinalEvaluationResult) -> None:
    baseline_for_csv = context.standalone_stats if context.standalone_stats is not None else context.reference_stats
    summary_rows = []
    if context.data_only_stats is not None:
        summary_rows.append({'step': -2, 'label': 'data_only', 'metrics': context.data_only_stats})
    if result.data_only_fcfw_stats is not None:
        summary_rows.append({'step': -3, 'label': 'data_only_fcfw', 'metrics': result.data_only_fcfw_stats})
    summary_rows.append({'step': -4, 'label': 'ensemble_final', 'metrics': result.final_stats})
    if result.ensemble_fcfw_stats is not None:
        summary_rows.append({'step': -5, 'label': 'ensemble_fcfw', 'metrics': result.ensemble_fcfw_stats})

    context.output.save_results_csv(
        context.ensemble_metrics_history,
        baseline_stats=baseline_for_csv,
        summary_rows=summary_rows,
    )


def _plot_final_progression(context: FinalEvaluationContext) -> None:
    if not get_plot_config()['plot_data_loss']:
        return

    plot_data_ensemble_loss(
        context.ensemble,
        context.ensemble_metrics_history,
        context.reference_stats,
        context.output,
        baseline_train_losses=(
            np.array(context.baseline_train_losses)
            if context.baseline_train_losses is not None
            else None
        ),
        data_only_history=context.data_only_history,
    )

    if context.evaluation.sampling_enabled:
        metric_configs = context.metric_configs
        if metric_configs is None:
            metric_configs = [
                ('mmd', 'Sampled MMD^2', 1, 'blue', 's'),
                ('tvd', 'TVD', 1, 'green', '^'),
                ('coverage', 'Coverage (%)', 100, 'purple', 'v'),
                ('validity', 'Validity (%)', 100, 'orange', 'd'),
            ]
        plot_metrics_progression(context.ensemble_metrics_history, context.reference_stats, context.output, metric_configs)


def _save_dataset_metrics(context: FinalEvaluationContext, result: FinalEvaluationResult) -> None:
    if not bool(context.config.get('dataset_metrics', True)):
        return
    if not context.evaluation.final_sampling_enabled or result.final_ensemble_samples is None:
        print("Skipping dataset metrics because final sampled outputs are unavailable.")
        return
    compute_and_save_dataset_metrics(
        dataset_bundle=context.dataset,
        output=context.output,
        final_samples=result.final_ensemble_samples,
        per_model_samples=result.per_model_samples,
    )


def _data_traces(ensemble: BoostedEnsemble, data: np.ndarray) -> list:
    traces = []
    sigmas = ensemble.sigma if hasattr(ensemble.sigma, '__iter__') else [ensemble.sigma]
    for sigma_idx in range(len(sigmas)):
        if sigma_idx in ensemble.terms.ops:
            _, visible_ops = ensemble.terms.ops[sigma_idx]
            trace = np.mean(1 - 2 * ((data @ np.asarray(visible_ops).T) % 2), axis=0)
            traces.append(trace)
    return traces


def _table_title(context: FinalEvaluationContext, result: FinalEvaluationResult) -> str:
    if context.evaluation.final_sampling_enabled:
        if result.ensemble_fcfw_stats is not None:
            return "FINAL MODEL COMPARISON (inc. FCFW)"
        return "FINAL MODEL COMPARISON"

    if result.ensemble_fcfw_stats is not None:
        return "FINAL MODEL COMPARISON (Analytical, inc. FCFW)"
    return "FINAL MODEL COMPARISON (Analytical)"


def _print_fcfw_metrics(label: str, mode_label: str, stats: dict) -> None:
    print(f"  FCFW {mode_label} MMD^2: {stats['mmd']:.6f}")
    if mode_label == "Sampled" and 'tvd' in stats and not np.isnan(stats['tvd']):
        print(f"  FCFW Sampled TVD:   {stats['tvd']:.4f}")


def _report_fcfw(context: FinalEvaluationContext) -> bool:
    return bool(context.config.get('report_fcfw', True))


def _final_eval_rng(context: FinalEvaluationContext) -> np.random.Generator:
    return np.random.default_rng(context.rng_seed + context.config['n_models'] * 7919)


def _analytical_stats(mmd: float) -> dict:
    return {
        'mmd': mmd,
        'kl': float('nan'),
        'jsd': float('nan'),
        'tvd': float('nan'),
        'validity': float('nan'),
        'coverage': float('nan'),
        'precision': float('nan'),
        'recall': float('nan'),
        'support_match': float('nan'),
        'f_score': float('nan'),
        'corr_fro': float('nan'),
    }
