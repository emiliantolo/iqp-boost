#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")/../.." && pwd)"

uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/calorimeter_5x4_b2th04_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/calorimeter_5x4_b2th06_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hamming_balls_5x4_k4p008_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hamming_balls_5x4_k4p012_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hamming_balls_5x4_k8p008_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hamming_balls_5x4_k8p012_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hopfield_5x4_p1b15_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hopfield_5x4_p1b20_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hopfield_5x4_p2b15_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hopfield_5x4_p2b20_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/ising_spin_glass_5x4_b10_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/ising_spin_glass_5x4_b15_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/ising_spin_glass_5x4_b20_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/topological_syndromes_5x4_p005_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/topological_syndromes_5x4_p010_grid2d.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/topological_syndromes_5x4_p015_grid2d.json"
