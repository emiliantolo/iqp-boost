# Experiments

This branch keeps Hopfield as the only runnable dataset family.

## Supported Datasets

- `hopfield`: available now through `configs/datasets/hopfield_16q_grid.json` and the Hopfield HPO configs in `configs/hpo/`.
- `hamming_ball`: reserved for the future `HammingBallDataset` implementation. Do not use it in configs until `src/datasets/hamming_ball.py` lands.

Historical benchmark and grid configs for removed datasets were deleted so stale experiments are not advertised as runnable.
