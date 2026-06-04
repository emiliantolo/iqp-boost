"""Weight strategy selection and mutation for boosted ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.core.ensemble import BoostedEnsemble
from src.core.weighting import (
    compute_optimal_alpha_coverage,
    compute_optimal_alpha_dual,
    compute_optimal_alpha_samples,
    compute_optimal_alpha_tvd_samples,
    compute_optimal_alpha_validity,
    compute_optimal_alpha_validity_coverage_sum,
    compute_optimal_weights_qp,
)


@dataclass
class WeightStrategyContext:
    ensemble: BoostedEnsemble
    strategy: str
    ground_truth: np.ndarray | None = None
    trs_old: list[np.ndarray] | None = None
    trs_new: list[np.ndarray] | None = None
    trs_data: list[np.ndarray] | None = None
    trs_corr_old: list[np.ndarray] | None = None
    trs_corr_new: list[np.ndarray] | None = None
    samples_old: np.ndarray | None = None
    samples_new: np.ndarray | None = None
    validity_fn: Callable | None = None
    coverage_fn: Callable | None = None
    alpha_n_grid: int = 11
    validity_weight: float = 0.5


@dataclass
class WeightStrategyResult:
    strategy: str
    alpha: float
    weights: np.ndarray
    fallback_to_greedy: bool = False
    warning: str | None = None


def apply_weight_strategy(context: WeightStrategyContext) -> WeightStrategyResult:
    strategy = context.strategy

    if strategy in ('greedy', 'frank_wolfe'):
        return _greedy_result(context, strategy)
    if strategy == 'line_search':
        return _line_search(context)
    if strategy == 'fully_corrective':
        return _fully_corrective(context)
    if strategy == 'tvd_line_search':
        return _tvd_line_search(context)
    if strategy == 'validity_line_search':
        return _validity_line_search(context)
    if strategy == 'coverage_line_search':
        return _coverage_line_search(context)
    if strategy == 'sum_line_search':
        return _sum_line_search(context)

    raise ValueError(f"Unknown weight_strategy: {strategy}")


def _greedy_result(context: WeightStrategyContext, strategy: str) -> WeightStrategyResult:
    alpha = _current_alpha(context.ensemble)
    print(f"  Weighting: {strategy} alpha = {alpha:.4f}")
    return _result(context, alpha=alpha)


def _line_search(context: WeightStrategyContext) -> WeightStrategyResult:
    if (
        context.samples_old is not None
        and context.samples_new is not None
        and context.ground_truth is not None
    ):
        alpha = compute_optimal_alpha_samples(
            context.samples_old,
            context.samples_new,
            context.ground_truth,
            context.ensemble.sigma,
        )
        print(f"  Weighting: sample line search alpha_opt = {alpha:.4f}")
        _apply_alpha(context.ensemble, alpha)
        return _result(context, alpha=alpha)

    if context.trs_old is not None and context.trs_new is not None and context.trs_data is not None:
        alpha = compute_optimal_alpha_dual(
            context.trs_old,
            context.trs_new,
            context.trs_data,
            trs_corr_old=context.trs_corr_old,
            trs_corr_new=context.trs_corr_new,
            n_samples=context.ensemble.n_samples,
        )
        print(f"  Weighting: dual line search alpha_opt = {alpha:.4f}")
        _apply_alpha(context.ensemble, alpha)
        return _result(context, alpha=alpha)

    return _fallback(context, "  [Warning] Missing data for line search. Falling back to greedy.")


def _fully_corrective(context: WeightStrategyContext) -> WeightStrategyResult:
    if context.trs_data is None:
        return _fallback(context, "  [Warning] Missing data traces for QP. Falling back to greedy.")

    weights = compute_optimal_weights_qp(
        all_trs=context.ensemble.terms.trs,
        trs_data=context.trs_data,
        all_corrs=context.ensemble.terms.corrs,
        n_samples=context.ensemble.n_samples,
    )
    context.ensemble.weights = np.asarray(weights, dtype=float).tolist()
    context.ensemble.normalize_weights()
    weights_text = ', '.join(f'{w:.4f}' for w in context.ensemble.weights)
    print(f"  Weighting: fully corrective QP weights = [{weights_text}]")
    return _result(context, alpha=_current_alpha(context.ensemble))


def _tvd_line_search(context: WeightStrategyContext) -> WeightStrategyResult:
    if context.samples_old is None or context.samples_new is None or context.ground_truth is None:
        return _fallback(context, "  [Warning] Missing samples/data for TVD line search. Falling back to greedy.")

    alpha = compute_optimal_alpha_tvd_samples(
        samples_old=context.samples_old,
        samples_new=context.samples_new,
        ground_truth=context.ground_truth,
    )
    print(f"  Weighting: TVD line search alpha_opt = {alpha:.4f}")
    _apply_alpha(context.ensemble, alpha)
    return _result(context, alpha=alpha)


def _validity_line_search(context: WeightStrategyContext) -> WeightStrategyResult:
    if context.samples_old is None or context.samples_new is None or context.validity_fn is None:
        return _fallback(context, "  [Warning] Missing samples/validity_fn for validity line search. Falling back to greedy.")

    alpha = compute_optimal_alpha_validity(
        samples_old=context.samples_old,
        samples_new=context.samples_new,
        validity_fn=context.validity_fn,
        n_grid=int(context.alpha_n_grid),
    )
    print(f"  Weighting: validity line search alpha_opt = {alpha:.4f}")
    _apply_alpha(context.ensemble, alpha)
    return _result(context, alpha=alpha)


def _coverage_line_search(context: WeightStrategyContext) -> WeightStrategyResult:
    if (
        context.samples_old is None
        or context.samples_new is None
        or context.ground_truth is None
        or context.coverage_fn is None
    ):
        return _fallback(context, "  [Warning] Missing samples/data/coverage_fn for coverage line search. Falling back to greedy.")

    alpha = compute_optimal_alpha_coverage(
        samples_old=context.samples_old,
        samples_new=context.samples_new,
        ground_truth=context.ground_truth,
        coverage_fn=context.coverage_fn,
        n_grid=int(context.alpha_n_grid),
    )
    print(f"  Weighting: coverage line search alpha_opt = {alpha:.4f}")
    _apply_alpha(context.ensemble, alpha)
    return _result(context, alpha=alpha)


def _sum_line_search(context: WeightStrategyContext) -> WeightStrategyResult:
    if (
        context.samples_old is None
        or context.samples_new is None
        or context.ground_truth is None
        or context.validity_fn is None
        or context.coverage_fn is None
    ):
        return _fallback(context, "  [Warning] Missing inputs for validity/coverage sum line search. Falling back to greedy.")

    alpha = compute_optimal_alpha_validity_coverage_sum(
        samples_old=context.samples_old,
        samples_new=context.samples_new,
        ground_truth=context.ground_truth,
        validity_fn=context.validity_fn,
        coverage_fn=context.coverage_fn,
        validity_weight=float(context.validity_weight),
        n_grid=int(context.alpha_n_grid),
    )
    print(f"  Weighting: validity/coverage sum line search alpha_opt = {alpha:.4f}")
    _apply_alpha(context.ensemble, alpha)
    return _result(context, alpha=alpha)


def _fallback(context: WeightStrategyContext, warning: str) -> WeightStrategyResult:
    print(warning)
    return _result(
        context,
        alpha=_current_alpha(context.ensemble),
        fallback_to_greedy=True,
        warning=warning,
    )


def _apply_alpha(ensemble: BoostedEnsemble, alpha: float) -> None:
    if len(ensemble.weights) < 2:
        return
    old_weight_sum = sum(ensemble.weights[:-1])
    if old_weight_sum > 0:
        scale = (1.0 - alpha) / old_weight_sum
        for idx in range(len(ensemble.weights) - 1):
            ensemble.weights[idx] *= scale
        ensemble.weights[-1] = alpha
    ensemble.normalize_weights()


def _current_alpha(ensemble: BoostedEnsemble) -> float:
    if not ensemble.weights:
        return float('nan')
    return float(ensemble.weights[-1])


def _result(
    context: WeightStrategyContext,
    *,
    alpha: float,
    fallback_to_greedy: bool = False,
    warning: str | None = None,
) -> WeightStrategyResult:
    return WeightStrategyResult(
        strategy=context.strategy,
        alpha=float(alpha),
        weights=np.asarray(context.ensemble.weights, dtype=np.float64),
        fallback_to_greedy=fallback_to_greedy,
        warning=warning,
    )
