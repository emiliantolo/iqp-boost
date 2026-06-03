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

### Baseline

The Experiment run comparison path that trains optional standalone and
data-only models, reports their metrics, and selects the reference used by
final evaluation.

### Final evaluation

The Experiment run phase that produces final ensemble metrics, comparison rows,
fully corrective Frank-Wolfe variants, held-out test metrics, plots, and CSV
summaries.

### Boosting step

One candidate model training/evaluation decision inside an Experiment run,
including weight selection, acceptance, metrics history, and diagnostics.

### Weight strategy

The Experiment run policy that selects or corrects ensemble mixture weights
from traces, samples, and dataset metric callbacks.
