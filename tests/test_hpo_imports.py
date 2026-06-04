import subprocess
import sys


def test_hpo_help_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "src.hpo", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in result.stdout


def test_experiment_factory_import_smoke():
    import src.experiments.factory  # noqa: F401
