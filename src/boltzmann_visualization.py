from __future__ import annotations

from .boltzmann_metrics import spin_covariance_from_samples, spin_covariance_from_probs
from .boltzmann_metrics import covariance_matrices
from .boltzmann_plots import (
    plot_hamming_weight_histogram,
    plot_covariance_heatmaps,
    plot_lorenz_curve,
)


def generate_boltzmann_visualizations(output_manager, x_train, baseline_samples,
                                      final_ensemble_samples, per_model_samples,
                                      weights, dataset):
    """Generate scalable Boltzmann diagnostics for a dataset run.

    The callback compares the baseline and final ensemble directly.
    """
    del weights

    if baseline_samples is None or final_ensemble_samples is None:
        print('Skipping Boltzmann visualizations because baseline or ensemble samples are unavailable.')
        return

    exact_probs = getattr(dataset, 'probs', None)

    if x_train is None or len(x_train) == 0:
        reference_samples = baseline_samples
    else:
        reference_samples = x_train

    fig = plot_hamming_weight_histogram(reference_samples, baseline_samples, final_ensemble_samples)
    path = output_manager.get_path('boltzmann_hamming_weights.png')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    fig.savefig(str(path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f'Saved Boltzmann hamming histogram to: {path}')

    if exact_probs is not None:
        sigma_reference = spin_covariance_from_probs(exact_probs)
    else:
        sigma_reference = spin_covariance_from_samples(reference_samples)
    sigma_baseline, sigma_ensemble = covariance_matrices(baseline_samples, final_ensemble_samples, exact_probs=exact_probs)
    fig = plot_covariance_heatmaps(sigma_reference, sigma_baseline, sigma_ensemble)
    path = output_manager.get_path('boltzmann_covariance_heatmaps.png')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    fig.savefig(str(path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f'Saved Boltzmann covariance heatmaps to: {path}')

    fig = plot_lorenz_curve(exact_probs, baseline_samples, final_ensemble_samples, reference_samples=x_train)
    path = output_manager.get_path('boltzmann_lorenz_curve.png')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    fig.savefig(str(path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f'Saved Boltzmann Lorenz curve to: {path}')
