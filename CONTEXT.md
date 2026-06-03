# iqp-boost Context

## Domain Terms

### Dataset catalog

The source of truth for supported config dataset keys, dataset defaults, sample
generation, exact probability support, dataset metric hooks, and dataset
visualization hooks.

### Dataset integration

A catalog-owned bundle of generated train/test samples, exact probabilities
when available, dataset metrics, and dataset visualization hooks. Experiment
runs consume dataset integrations instead of reconstructing dataset-specific
capabilities from scattered kwargs.

### Experiment run

One config-resolved IQP boosting execution over a dataset integration. A run
owns circuit setup, sigma selection, baseline training, ensemble boosting,
evaluation, reporting, and output for that resolved config.

### Final evaluation

The Experiment run phase that produces final ensemble metrics, comparison rows,
fully corrective Frank-Wolfe variants, held-out test metrics, plots, and CSV
summaries.
