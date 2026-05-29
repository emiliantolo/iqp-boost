# HPO Experiment Configurations: Hopfield Dataset (20 Qubits)

This document outlines the proposed dataset configurations and the theoretical justification for benchmarking **boosted IQP ensembles** on the 20-qubit Hopfield network.

## Target Configurations

We focus on the HPO of the IQP boosting framework for a **20-qubit Hopfield dataset** ($2^{20} \approx 10^6$ states) across the following grid:

*   **Inverse Temperature ($\beta$)**: `[1.5, 2.0]`
*   **Number of Patterns ($P$)**: `[1, 2]`

---

## Dataset Configuration Justifications

| Configuration | Landscape Type | Generative Modeling Justification | Boosting-Specific Justification |
| :--- | :--- | :--- | :--- |
| **$P=1, \beta=1.5$** | Soft Bimodal | **Ferromagnetic Baseline**: Two broad, symmetric global energy wells at the stored pattern $p$ and its negation $-p$. Significant thermal leak into neighboring Hamming states. | **Control Case**: Verifies if a single base IQP circuit can easily model symmetric bimodal landscapes and soft local correlations. |
| **$P=1, \beta=2.0$** | Sharp Bimodal | **Mode Concentration**: Two extremely narrow, delta-like peak distributions at $\pm p$. Zero transition probability between wells. | **Peak Sensitivity**: Tests the optimizer's resilience against gradient sparsity (vanishing gradients) when modes are highly localized in discrete space. |
| **$P=2, \beta=1.5$** | Soft Multimodal | **Spurious Mode Clustering**: 4 primary modes ($\pm p^1, \pm p^2$) and several local mixture/spurious states with non-trivial overlap and soft transition states. | **Mass Allocation**: Evaluates the model's ability to smoothly partition probability mass across multiple overlapping clusters without underfitting. |
| **$P=2, \beta=2.0$** | Sharp Multimodal | **Extreme Mode Collapse Test**: 4 to 8 disjoint, highly concentrated modes separated by deep, forbidden energy barriers. | **Repulsion & Specialization**: **The ultimate test for dual MMD boosting.** Tests whether ensemble repulsion successfully forces sub-models to specialize on different peaks rather than collapsing to a single mode. |

---

## Algorithmic Connections to Boosting

### 1. Functional Coordinate Descent via the Witness Function
Under the dual MMD framework with $\lambda_{\text{dual}} = 1.0$, the loss function for a new candidate model $P_T$ is governed by the **witness function** $W(x)$:
$$W(x) = K(x, E) - K(x, D)$$
where $E$ is the current ensemble and $D$ is the target data distribution.
*   **Attraction ($W(x) < 0$)**: Occurs where the target Hopfield modes have high mass, but the current ensemble has placed none.
*   **Repulsion ($W(x) > 0$)**: Occurs where the ensemble has already placed sufficient mass.
*   **Significance**: For $P=2, \beta=2.0$, this force successfully repels new models from already covered patterns (e.g., $\pm p^1$), driving them to discover and specialize on the remaining uncovered patterns (e.g., $\pm p^2$).

### 2. Physical Temperature vs. Kernel Scale ($\beta$ vs. $\sigma$)
The Gaussian kernel bandwidth $\sigma$ defines the resolution of the repulsion force:
$$k(x, y) = \exp\left( -\frac{d(x, y)^2}{2\sigma^2} \right)$$
*   For **high $\beta$ (sharp peaks)**, the target states are highly localized. If the bandwidth $\sigma$ is too large, the repulsion from $\pm p^1$ will overflow and poison the learning of $\pm p^2$. If it is too small, gradient signal is lost.
*   **HPO Objective**: Optimizing the relationship between the dataset inverse temperature $\beta$ and the kernel bandwidth $\sigma$ is a primary hyperparameter goal.

### 3. Mode Cardinality vs. Weighting Strategy
With $P=2$, the target distribution has exactly 4 major symmetric peaks.
*   An ensemble of $M \ge 4$ models has sufficient capacity to cover all peaks.
*   Instead of rigid step-size updates (e.g., Frank-Wolfe $\alpha_t = \frac{2}{t+2}$), this discrete structure is highly suited for **Fully Corrective Frank-Wolfe (FCFW)** or **line-search weight optimization**, which can dynamically assign $50\%\text{--}50\%$ weights to specialized sub-models in very few boosting steps.
