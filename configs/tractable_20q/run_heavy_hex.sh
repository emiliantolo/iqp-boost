#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")/../.." && pwd)"

uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/calorimeter_5x4_b2th04_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/calorimeter_5x4_b2th06_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hamming_balls_5x4_k4p008_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hamming_balls_5x4_k4p012_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hamming_balls_5x4_k8p008_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hamming_balls_5x4_k8p012_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hopfield_5x4_p1b15_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hopfield_5x4_p1b20_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hopfield_5x4_p2b15_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/hopfield_5x4_p2b20_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/ising_spin_glass_5x4_b10_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/ising_spin_glass_5x4_b15_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/ising_spin_glass_5x4_b20_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/topological_syndromes_5x4_p005_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/topological_syndromes_5x4_p010_heavy_hex.json"
uv run python "$DIR/main.py" --config "$DIR/configs/tractable_20q/topological_syndromes_5x4_p015_heavy_hex.json"
