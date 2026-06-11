#!/usr/bin/env bash
set -euo pipefail

# Recommended CPU opts (from README)
#export XLA_FLAGS="--xla_cpu_max_isa=AVX512 --xla_cpu_enable_fast_math=true"
#export XLA_PYTHON_CLIENT_PREALLOCATE=false

HPO_DIR="configs/hpo"

run_and_commit() {
  local config="$1"
  local name
  name=$(basename "$config" .json)
  echo ">>> $name"
  uv run python3 -m src.hpo --config "$config"

  # Find the latest study directory just created
  local study_dir
  study_dir=$(ls -td out/hpo/"${name}"_* 2>/dev/null | head -n 1)

  if [ -n "$study_dir" ] && [ -d "$study_dir" ]; then
    echo "  Committing plots from $study_dir"
    # Find all plot files (pdf, png) in the study directory and add them
    find "$study_dir" -type f \( -name "*.pdf" -o -name "*.png" \) -print0 | while IFS= read -r -d '' plot; do
      git add "$plot"
    done
    git commit -m "hpo: $name plots" || echo "  No new plots to commit"
  else
    echo "  No study directory found, skipping commit"
  fi

  git push
}

echo "=== 20q Hamming Balls — heavy hex topology ==="
for f in "$HPO_DIR"/hamming_balls/heavy_hex/hamming_balls_20q_*.json; do
  run_and_commit "$f"
done

echo "=== 20q Hopfield — heavy hex topology ==="
for f in "$HPO_DIR"/hopfield/heavy_hex/hopfield_20q_*.json; do
  run_and_commit "$f"
done

echo "=== 20q Hamming Balls — grid 2D topology ==="
for f in "$HPO_DIR"/hamming_balls/grid_2d/hamming_balls_20q_*.json; do
  run_and_commit "$f"
done

echo "=== 20q Hopfield — grid 2D topology ==="
for f in "$HPO_DIR"/hopfield/grid_2d/hopfield_20q_*.json; do
  run_and_commit "$f"
done

echo "=== 20q Hamming Balls — local topology ==="
for f in "$HPO_DIR"/hamming_balls/local/hamming_balls_20q_*.json; do
  run_and_commit "$f"
done

echo "=== 20q Hopfield — local topology ==="
for f in "$HPO_DIR"/hopfield/local/hopfield_20q_*.json; do
  run_and_commit "$f"
done

echo "=== 100q MNIST — all classes ==="
for f in "$HPO_DIR"/mnist/mnist_100q_*.json; do
  run_and_commit "$f"
done

echo "=== All HPO studies complete ==="
