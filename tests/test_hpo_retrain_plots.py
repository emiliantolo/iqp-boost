from src.hpo.retrain_plots import plot_best_retrain_summary


def test_plot_best_retrain_summary_writes_comparison_plots(tmp_path):
    summary = {
        "seeds": [
            {
                "seed_index": 0,
                "final_stats": {"mmd": 0.1, "test_mmd": 0.2},
                "baseline_stats": {"mmd": 0.3, "test_mmd": 0.4},
                "ensemble_fcfw_stats": {"mmd": 0.08, "test_mmd": 0.18},
                "weights": [0.25, 0.75],
                "ensemble_fcfw_weights": [0.4, 0.6],
            },
            {
                "seed_index": 1,
                "final_stats": {"mmd": 0.2, "test_mmd": 0.3},
                "baseline_stats": {"mmd": 0.5, "test_mmd": 0.6},
                "ensemble_fcfw_stats": {"mmd": 0.18, "test_mmd": 0.28},
                "weights": [0.5, 0.5],
                "ensemble_fcfw_weights": [0.55, 0.45],
            },
        ],
        "aggregates": {
            "final_stats": {
                "mmd": {"mean": 0.15, "std": 0.05},
                "test_mmd": {"mean": 0.25, "std": 0.05},
            },
            "baseline_stats": {
                "mmd": {"mean": 0.4, "std": 0.1},
                "test_mmd": {"mean": 0.5, "std": 0.1},
            },
            "ensemble_fcfw_stats": {
                "mmd": {"mean": 0.13, "std": 0.05},
                "test_mmd": {"mean": 0.23, "std": 0.05},
            },
        },
    }

    paths = plot_best_retrain_summary(summary, tmp_path)

    assert (tmp_path / "plots" / "metrics_comparison.pdf").exists()
    assert (tmp_path / "plots" / "metrics_comparison.png").exists()
    assert (tmp_path / "plots" / "metric_trends.pdf").exists()
    assert (tmp_path / "plots" / "metric_trends.png").exists()
    assert (tmp_path / "plots" / "weight_distribution.pdf").exists()
    assert (tmp_path / "plots" / "weight_distribution.png").exists()
    assert "metrics_comparison_pdf" in paths


def test_plot_best_retrain_summary_can_write_mmd_only_plots(tmp_path):
    summary = {
        "seeds": [
            {
                "seed_index": 0,
                "final_stats": {"mmd": 0.1, "test_mmd": 0.2, "tvd": 0.9},
                "baseline_stats": {"mmd": 0.3, "test_mmd": 0.4, "tvd": 0.8},
                "ensemble_fcfw_stats": {"mmd": 0.08, "test_mmd": 0.18, "tvd": 0.7},
                "weights": [0.25, 0.75],
                "ensemble_fcfw_weights": [0.4, 0.6],
            }
        ],
        "aggregates": {
            "final_stats": {
                "mmd": {"mean": 0.1, "std": 0.0},
                "test_mmd": {"mean": 0.2, "std": 0.0},
                "tvd": {"mean": 0.9, "std": 0.0},
            },
            "baseline_stats": {
                "mmd": {"mean": 0.3, "std": 0.0},
                "test_mmd": {"mean": 0.4, "std": 0.0},
                "tvd": {"mean": 0.8, "std": 0.0},
            },
            "ensemble_fcfw_stats": {
                "mmd": {"mean": 0.08, "std": 0.0},
                "test_mmd": {"mean": 0.18, "std": 0.0},
                "tvd": {"mean": 0.7, "std": 0.0},
            },
        },
    }

    paths = plot_best_retrain_summary(
        summary,
        tmp_path,
        metric_filter="mmd",
        include_weight_distribution=False,
    )

    assert (tmp_path / "plots" / "metrics_comparison.pdf").exists()
    assert (tmp_path / "plots" / "metric_trends.pdf").exists()
    assert not (tmp_path / "plots" / "weight_distribution.pdf").exists()
    assert "metrics_comparison_pdf" in paths
    assert "weight_distribution_pdf" not in paths
