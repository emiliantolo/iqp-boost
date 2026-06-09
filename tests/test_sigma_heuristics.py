import numpy as np

from src.core.sigma_heuristics import compute_sigma


def test_explicit_sigma_takes_precedence_over_legacy_sigma_factor(capsys):
    x_train = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )
    config = {
        "sigma": [5.0, 2.5, 1.5, 1.0],
        "sigma_factor": [0.5],
    }

    assert compute_sigma(config, x_train, seed=7) == [5.0, 2.5, 1.5, 1.0]
    assert "[sigma:median]" not in capsys.readouterr().out
