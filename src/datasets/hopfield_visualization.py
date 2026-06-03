"""Visualization utilities for Hopfield dataset integrations."""

import json
import numpy as np
import matplotlib.pyplot as plt

def samples_to_probs(samples, n_qubits):
    """Convert binary samples in {0, 1} to empirical probabilities of length 2^N."""
    if samples is None or len(samples) == 0:
        return np.zeros(2**n_qubits)
    # Cast to float64 to avoid int8 overflow during matrix multiplication/sum
    indices = np.sum(samples.astype(np.float64) * (2 ** np.arange(n_qubits)), axis=1).astype(int)
    counts = np.bincount(indices, minlength=2**n_qubits)
    return counts / counts.sum()

def probs_to_grid(probs, n_qubits):
    """Reshape a 1D probability array of size 2^N into a 2^N1 x 2^N2 grid."""
    n2 = n_qubits // 2
    n1 = n_qubits - n2
    return probs.reshape(2**n2, 2**n1).T

def compute_spin_correlation_matrix(probs, n_qubits):
    """Compute exact spin correlation matrix C_ij = <s_i s_j> from probabilities."""
    indices = np.arange(2**n_qubits)
    bit_positions = np.arange(n_qubits)
    x = ((indices[:, None] >> bit_positions) & 1)
    S = 1.0 - 2.0 * x
    return (S.T * probs) @ S

def compute_empirical_spin_correlation_matrix(samples):
    """Compute empirical spin correlation matrix C_ij = <s_i s_j> from samples."""
    if samples is None or len(samples) == 0:
        return None
    # S_emp has shape (M, N)
    S_emp = 1.0 - 2.0 * samples.astype(np.float64)
    return (S_emp.T @ S_emp) / len(samples)

def get_binary_labels(n_bits):
    """Generate list of binary strings of length n_bits."""
    return [format(i, f'0{n_bits}b') for i in range(2**n_bits)]

def generate_hopfield_visualizations(output_manager, x_train, baseline_samples,
                                     final_ensemble_samples, per_model_samples,
                                     weights, dataset):
    """
    Main visualization generator for the Hopfield dataset.
    Creates:
      1. hopfield_probability_landscapes.png (heatmap comparison of state space)
      2. hopfield_spin_correlations.png (correlation heatmaps compared to Hebbian coupling J)
      3. hopfield_top_bitstrings.png (grouped bar chart comparing top states)
      4. hopfield_interactive.html (premium Plotly-based HTML dashboard)
    """
    n_qubits = dataset.n_qubits
    exact_probs = np.asarray(dataset.probs, dtype=np.float64)
    if exact_probs.size == 0 or not np.isfinite(exact_probs).any():
        print(
            f"Skipping Hopfield state-space visualizations for {n_qubits} qubits "
            "because exact probabilities are unavailable in MCMC mode."
        )
        return
    exact_probs = exact_probs / exact_probs.sum()

    # Calculate empirical probabilities
    baseline_probs = samples_to_probs(baseline_samples, n_qubits) if baseline_samples is not None else np.zeros_like(exact_probs)
    ensemble_probs = samples_to_probs(final_ensemble_samples, n_qubits) if final_ensemble_samples is not None else np.zeros_like(exact_probs)
    
    model_probs_list = []
    for m_samples in per_model_samples:
        model_probs_list.append(samples_to_probs(m_samples, n_qubits))

    # Compute correlation matrices
    C_exact = compute_spin_correlation_matrix(exact_probs, n_qubits)
    C_base = compute_empirical_spin_correlation_matrix(baseline_samples)
    C_ens = compute_empirical_spin_correlation_matrix(final_ensemble_samples)
    J = np.asarray(dataset.J, dtype=np.float64)

    # Save Static Plots
    _save_probability_landscapes(output_manager, exact_probs, baseline_probs, ensemble_probs, model_probs_list, n_qubits)
    _save_spin_correlations(output_manager, C_exact, C_base, C_ens, J, n_qubits)
    _save_top_bitstrings(output_manager, exact_probs, baseline_probs, ensemble_probs, model_probs_list, n_qubits)
    
    # Save Interactive Dashboard
    _save_interactive_dashboard(output_manager, exact_probs, baseline_probs, ensemble_probs, model_probs_list, weights, dataset, C_exact, C_base, C_ens)

    # Save Raw Data JSON and LaTeX table
    _save_raw_data(output_manager, exact_probs, baseline_probs, ensemble_probs, model_probs_list, weights, dataset, C_exact, C_base, C_ens, J)
    _save_latex_table(output_manager, exact_probs, baseline_probs, ensemble_probs, n_qubits)

def _save_probability_landscapes(output_manager, exact_probs, baseline_probs, ensemble_probs, model_probs_list, n_qubits):
    """Plot side-by-side heatmaps of the state space distribution."""
    # We display: Ground Truth, Baseline, Combined Ensemble, and up to 3 individual models
    n_models_to_show = min(len(model_probs_list), 3)
    n_plots = 3 + n_models_to_show
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else np.array([axes])

    plot_configs = [
        ("Ground Truth (Exact)", exact_probs, "plasma"),
        ("Baseline (Sampled)", baseline_probs, "plasma"),
        ("Combined Ensemble", ensemble_probs, "plasma")
    ]
    for idx, probs in enumerate(model_probs_list[:n_models_to_show]):
        plot_configs.append((f"Model {idx} (Sampled)", probs, "plasma"))

    n2 = n_qubits // 2
    n1 = n_qubits - n2

    for idx, (title, probs, cmap) in enumerate(plot_configs):
        ax = axes[idx]
        grid = probs_to_grid(probs, n_qubits)
        
        im = ax.imshow(grid, cmap=cmap, origin='lower', aspect='auto')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # Grid settings
        if n_qubits <= 8:
            ax.set_xticks(np.arange(2**n1))
            ax.set_yticks(np.arange(2**n2))
            ax.set_xticklabels(get_binary_labels(n1), rotation=45, fontsize=8)
            ax.set_yticklabels(get_binary_labels(n2), fontsize=8)
        else:
            # Turn off ticks for larger qubit counts to avoid clutter
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"Lower {n1} qubits (binary space)")
            ax.set_ylabel(f"Upper {n2} qubits (binary space)")

        fig.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused plots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Hopfield Probability Landscapes ({n_qubits} qubits)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    path = output_manager.get_path("hopfield_probability_landscapes.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(str(path).replace(".png", ".pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved probability landscapes to: {path}")

def _save_spin_correlations(output_manager, C_exact, C_base, C_ens, J, n_qubits):
    """Plot spin correlation matrices comparison to target coupling matrix J."""
    # We display: Coupling Matrix J, Ground Truth Correlations, Baseline Correlations, Ensemble Correlations
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    matrices = [
        ("Hebbian Coupling Matrix J", J),
        ("Exact Boltzmann Correlations", C_exact),
        ("Baseline Sampled Correlations", C_base),
        ("Ensemble Sampled Correlations", C_ens)
    ]

    for idx, (title, matrix) in enumerate(matrices):
        ax = axes[idx]
        if matrix is None:
            ax.text(0.5, 0.5, "Data unavailable", ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=11, fontweight='bold')
            continue

        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin='lower')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        
        ax.set_xticks(np.arange(n_qubits))
        ax.set_yticks(np.arange(n_qubits))
        ax.set_xticklabels(np.arange(n_qubits))
        ax.set_yticklabels(np.arange(n_qubits))
        ax.set_xlabel("Spin Index i")
        ax.set_ylabel("Spin Index j")
        
        # Overlay values if small number of qubits
        if n_qubits <= 8:
            for i in range(n_qubits):
                for j in range(n_qubits):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', 
                            color='black' if abs(matrix[i, j]) < 0.4 else 'white', fontsize=8)
        
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Hopfield Spin-Spin Correlations ({n_qubits} qubits)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    path = output_manager.get_path("hopfield_spin_correlations.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(str(path).replace(".png", ".pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved spin correlations to: {path}")

def _save_top_bitstrings(output_manager, exact_probs, baseline_probs, ensemble_probs, model_probs_list, n_qubits):
    """Plot horizontal grouped bar chart for top most probable states."""
    # Get top 15 states of Ground Truth
    top_k = min(15, 2**n_qubits)
    top_k_indices = np.argsort(exact_probs)[::-1][:top_k]
    
    # Binary string labels
    labels = [format(idx, f'0{n_qubits}b') for idx in top_k_indices]
    
    # Grab values
    exact_vals = exact_probs[top_k_indices]
    base_vals = baseline_probs[top_k_indices]
    ens_vals = ensemble_probs[top_k_indices]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(top_k)
    height = 0.25

    # Grouped bars
    ax.barh(y_pos + height, exact_vals, height, label='Exact Boltzmann', color='#1f77b4', alpha=0.9)
    ax.barh(y_pos, ens_vals, height, label='Combined Ensemble', color='#2ca02c', alpha=0.9)
    ax.barh(y_pos - height, base_vals, height, label='Baseline', color='#7f7f7f', alpha=0.7)

    # Plot individual model probabilities as scatter points overlaying the combined ensemble bars
    if len(model_probs_list) > 0:
        for m_idx, m_probs in enumerate(model_probs_list):
            m_vals = m_probs[top_k_indices]
            # Add slight jitter to y positions so dots don't overlap completely
            jitter = (m_idx - (len(model_probs_list) - 1) / 2) * 0.04
            ax.scatter(m_vals, y_pos + jitter, color='#ff7f0e', alpha=0.6, s=15, 
                       zorder=5, label='Individual Models' if m_idx == 0 else "")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, family='monospace', fontsize=9)
    ax.invert_yaxis()  # top-down
    ax.set_xlabel('Probability', fontsize=11, fontweight='bold')
    ax.set_ylabel('Bitstrings (State x)', fontsize=11, fontweight='bold')
    ax.set_title(f"Top {top_k} Most Probable States", fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=10)

    fig.suptitle(f"Hopfield Top Bitstrings Comparison ({n_qubits} qubits)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    path = output_manager.get_path("hopfield_top_bitstrings.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(str(path).replace(".png", ".pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved top bitstrings visualization to: {path}")


def _save_interactive_dashboard(output_manager, exact_probs, baseline_probs, ensemble_probs, model_probs_list, weights, dataset, C_exact, C_base, C_ens):
    """Generate and write a premium Plotly-based HTML dashboard."""
    n_qubits = dataset.n_qubits
    n2 = n_qubits // 2
    n1 = n_qubits - n2

    # Map states to string labels for plot tooltips
    all_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    
    # 2D Grid coordinates for heatmap/3D
    x_coords = get_binary_labels(n1)
    y_coords = get_binary_labels(n2)
    
    # Construct 2D grid arrays for Ground Truth, Baseline, Ensemble
    z_exact = probs_to_grid(exact_probs, n_qubits).T.tolist()  # (n2, n1) list
    z_base = probs_to_grid(baseline_probs, n_qubits).T.tolist()
    z_ens = probs_to_grid(ensemble_probs, n_qubits).T.tolist()

    # Get top 20 state indices
    top_k = min(20, 2**n_qubits)
    top_k_indices = np.argsort(exact_probs)[::-1][:top_k].tolist()
    top_k_labels = [format(idx, f'0{n_qubits}b') for idx in top_k_indices]
    
    top_exact_vals = exact_probs[top_k_indices].tolist()
    top_base_vals = baseline_probs[top_k_indices].tolist()
    top_ens_vals = ensemble_probs[top_k_indices].tolist()

    top_models_vals = []
    for m_probs in model_probs_list:
        top_models_vals.append(m_probs[top_k_indices].tolist())

    # Patterns
    patterns_list = dataset.patterns.tolist() if hasattr(dataset, 'patterns') else []
    
    # Weights
    weights_list = [float(w) for w in weights] if weights is not None else []

    # Matrix data to JSON
    J_list = dataset.J.tolist()
    C_exact_list = C_exact.tolist()
    C_base_list = C_base.tolist() if C_base is not None else []
    C_ens_list = C_ens.tolist() if C_ens is not None else []

    data_payload = {
        'nQubits': n_qubits,
        'n1': n1,
        'n2': n2,
        'xCoords': x_coords,
        'yCoords': y_coords,
        'zExact': z_exact,
        'zBase': z_base,
        'zEns': z_ens,
        'topLabels': top_k_labels,
        'topExact': top_exact_vals,
        'topBase': top_base_vals,
        'topEns': top_ens_vals,
        'topModels': top_models_vals,
        'weights': weights_list,
        'patterns': patterns_list,
        'J': J_list,
        'C_exact': C_exact_list,
        'C_base': C_base_list,
        'C_ens': C_ens_list,
        'beta': float(dataset.beta) if hasattr(dataset, 'beta') else 2.0
    }

    payload_json = json.dumps(data_payload)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hopfield Ensemble Boosting Distribution Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        :root {{
            --bg-color: #0f0c1b;
            --container-bg: rgba(22, 19, 45, 0.6);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #f3f0ff;
            --accent-primary: #9d4edd;
            --accent-secondary: #3a0ca3;
            --accent-green: #06d6a0;
            --accent-orange: #f77f00;
        }}
        
        body {{
            background-color: var(--bg-color);
            background-image: radial-gradient(circle at 10% 20%, rgba(90, 24, 154, 0.15) 0%, transparent 40%),
                              radial-gradient(circle at 90% 80%, rgba(58, 12, 163, 0.15) 0%, transparent 40%);
            color: var(--text-color);
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 0;
            min-height: 100vh;
        }}

        header {{
            padding: 2.5rem 3rem 1.5rem 3rem;
            border-bottom: 1px solid var(--border-color);
            background: rgba(15, 12, 27, 0.85);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        .header-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            margin: 0;
            font-weight: 800;
            font-size: 2.2rem;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #e0aaff, #9d4edd, #5a189a);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .meta-stats {{
            display: flex;
            gap: 1.5rem;
        }}

        .meta-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            color: #d8bbff;
        }}

        .meta-card strong {{
            color: var(--text-color);
            font-weight: 600;
        }}

        .dashboard-container {{
            max-width: 1450px;
            margin: 0 auto;
            padding: 2rem 3rem;
            display: grid;
            grid-template-columns: 1fr;
            gap: 2.5rem;
        }}

        .glass-card {{
            background: var(--container-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .glass-card:hover {{
            box-shadow: 0 8px 32px 0 rgba(157, 78, 221, 0.1);
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.8rem;
        }}

        .card-title {{
            font-weight: 600;
            font-size: 1.4rem;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}

        .card-subtitle {{
            font-size: 0.9rem;
            color: #b39ddb;
            margin-top: 0.2rem;
        }}

        /* Grid for multiple plots */
        .twin-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}

        @media (max-width: 1100px) {{
            .twin-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .controls {{
            display: flex;
            gap: 0.8rem;
            align-items: center;
        }}

        select {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 0.5rem 1.5rem;
            border-radius: 8px;
            outline: none;
            font-family: inherit;
            cursor: pointer;
            transition: all 0.3s;
        }}

        select:hover {{
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--accent-primary);
        }}

        .chart-div {{
            min-height: 480px;
            width: 100%;
        }}

        .pattern-chip-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 0.5rem;
        }}

        .pattern-chip {{
            background: rgba(157, 78, 221, 0.12);
            border: 1px solid rgba(157, 78, 221, 0.3);
            color: #e0aaff;
            font-family: 'JetBrains Mono', monospace;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .chip-label {{
            opacity: 0.6;
            font-size: 0.7rem;
            text-transform: uppercase;
        }}

        footer {{
            text-align: center;
            padding: 3rem;
            border-top: 1px solid var(--border-color);
            color: rgba(255, 255, 255, 0.3);
            font-size: 0.9rem;
            margin-top: 4rem;
        }}
    </style>
</head>
<body>
    <header>
        <div class="header-container">
            <div>
                <h1>Hopfield Distribution Analysis</h1>
                <div style="font-size:0.95rem; color:rgba(255,255,255,0.6); margin-top:0.3rem;">
                    Quantum IQP Boosting Ensemble Verification Dashboard
                </div>
            </div>
            <div class="meta-stats">
                <div class="meta-card">Qubits: <strong>{n_qubits}</strong></div>
                <div class="meta-card">Beta: <strong>{float(dataset.beta)}</strong></div>
                <div class="meta-card">Ensemble Size: <strong>{len(model_probs_list)} models</strong></div>
            </div>
        </div>
    </header>

    <div class="dashboard-container">
        <!-- 3D & 2D Landscape comparison -->
        <div class="glass-card">
            <div class="card-header">
                <div>
                    <h2 class="card-title">State Space Probability Landscape (3D / 2D)</h2>
                    <div class="card-subtitle">Showing probability landscape indexed over split qubit bitstrings</div>
                </div>
                <div class="controls">
                    <label for="landscape-select">Dataset:</label>
                    <select id="landscape-select" onchange="updateLandscapePlot()">
                        <option value="exact">Ground Truth (Exact)</option>
                        <option value="ensemble">Combined Ensemble</option>
                        <option value="baseline">Baseline (Sampled)</option>
                    </select>
                    <label for="plot-type-select" style="margin-left:15px;">View:</label>
                    <select id="plot-type-select" onchange="updateLandscapePlot()">
                        <option value="surface">3D Surface</option>
                        <option value="heatmap">2D Heatmap</option>
                    </select>
                </div>
            </div>
            <div id="landscape-div" class="chart-div"></div>
        </div>

        <div class="twin-grid">
            <!-- Top Bitstrings comparative -->
            <div class="glass-card">
                <div class="card-header">
                    <div>
                        <h2 class="card-title">Top State Comparison</h2>
                        <div class="card-subtitle">Ground Truth vs Ensemble vs Baseline for top 20 most probable states</div>
                    </div>
                </div>
                <div id="top-bar-div" class="chart-div"></div>
            </div>

            <!-- Stored memories info -->
            <div class="glass-card">
                <div class="card-header">
                    <div>
                        <h2 class="card-title">Stored Memories & Hebbian Couplings</h2>
                        <div class="card-subtitle">The Hebbian patterns encoded in the Hopfield network definition</div>
                    </div>
                </div>
                <div>
                    <h3 style="font-size:1.05rem; margin-bottom:0.8rem; color:#b585ff;">Encoded Spin Patterns (Memories):</h3>
                    <div class="pattern-chip-container" id="patterns-container"></div>

                    <h3 style="font-size:1.05rem; margin-top:2.5rem; margin-bottom:0.8rem; color:#b585ff;">Ensemble Weights per Component:</h3>
                    <div class="pattern-chip-container" id="weights-container"></div>
                    
                    <div style="margin-top:2rem; font-size:0.95rem; line-height:1.6; color:rgba(255,255,255,0.75);">
                        <p><strong>Note on RKHS Boosting:</strong> The ensemble model adds IQP components greedily to minimize the MMD distance.
                        The plot on the left compares the probabilities for the top states. Check how the Combined Ensemble (green)
                        is capable of capturing the high-probability Boltzmann modes compared to the covariance/standalone baseline.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Spin-Spin correlation plots -->
        <div class="glass-card">
            <div class="card-header">
                <div>
                    <h2 class="card-title">Spin-Spin Correlation Matrices &Couplings</h2>
                    <div class="card-subtitle">Comparing Hebbian coupling parameters J with exact and sampled spin correlations &lt;s_i s_j&gt;</div>
                </div>
                <div class="controls">
                    <label for="correlation-select">Select Matrix to inspect:</label>
                    <select id="correlation-select" onchange="drawCorrelationHeatmap()">
                        <option value="J">Hebbian Coupling Matrix J</option>
                        <option value="exact">Exact Boltzmann Correlations</option>
                        <option value="ensemble">Ensemble Sampled Correlations</option>
                        <option value="baseline">Baseline Sampled Correlations</option>
                    </select>
                </div>
            </div>
            <div id="correlation-div" class="chart-div" style="min-height:550px;"></div>
        </div>
    </div>

    <footer>
        DeepMind Advanced Agentic Coding - Antigravity Agent &copy; 2026
    </footer>

    <script>
        // Load data embedded by Python script
        const data = {payload_json};
        
        // Render Stored memories chips
        const pContainer = document.getElementById("patterns-container");
        data.patterns.forEach((pat, idx) => {{
            const chip = document.createElement("div");
            chip.className = "pattern-chip";
            
            // Format pat as spins
            const spinStr = pat.map(val => val > 0 ? "+" : "-").join(" ");
            chip.innerHTML = `<span class="chip-label">Pattern ${{idx+1}}:</span> [ ${{spinStr}} ]`;
            pContainer.appendChild(chip);
        }});

        // Render Ensemble weights chips
        const wContainer = document.getElementById("weights-container");
        if (data.weights.length > 0) {{
            data.weights.forEach((w, idx) => {{
                const chip = document.createElement("div");
                chip.className = "pattern-chip";
                chip.style.borderColor = "rgba(6, 214, 160, 0.4)";
                chip.style.color = "#a3f7bf";
                chip.innerHTML = `<span class="chip-label" style="color:#06d6a0;">Model ${{idx}}:</span> w = ${{w.toFixed(4)}}`;
                wContainer.appendChild(chip);
            }});
        }} else {{
            wContainer.innerHTML = '<div style="color:rgba(255,255,255,0.4);">No models trained in the ensemble yet (Step 0)</div>';
        }}

        // Layout presets
        const commonLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{
                family: 'Outfit, sans-serif',
                color: '#f3f0ff'
            }},
            margin: {{t: 30, r: 30, b: 60, l: 60}}
        }};

        // Render Top states chart
        function drawTopBarChart() {{
            const exactTrace = {{
                x: data.topLabels,
                y: data.topExact,
                name: 'Exact Boltzmann',
                type: 'bar',
                marker: {{color: '#9d4edd', opacity: 0.85}}
            }};
            
            const ensTrace = {{
                x: data.topLabels,
                y: data.topEns,
                name: 'Combined Ensemble',
                type: 'bar',
                marker: {{color: '#06d6a0', opacity: 0.85}}
            }};

            const baseTrace = {{
                x: data.topLabels,
                y: data.topBase,
                name: 'Baseline',
                type: 'bar',
                marker: {{color: '#7f7f7f', opacity: 0.6}}
            }};

            const traces = [exactTrace, ensTrace, baseTrace];
            
            // Add individual models if small number of models
            if (data.topModels && data.topModels.length > 0 && data.topModels.length <= 6) {{
                data.topModels.forEach((m_vals, idx) => {{
                    traces.push({{
                        x: data.topLabels,
                        y: m_vals,
                        name: `Model ${{idx}}`,
                        type: 'scatter',
                        mode: 'markers',
                        marker: {{size: 7, opacity: 0.5, symbol: 'circle'}}
                    }});
                }});
            }}

            const layout = {{
                ...commonLayout,
                barmode: 'group',
                xaxis: {{
                    title: 'Bitstrings (monospace state configurations)',
                    tickangle: 45,
                    font: {{family: 'JetBrains Mono, monospace'}}
                }},
                yaxis: {{
                    title: 'Probability (PDF)',
                    gridcolor: 'rgba(255,255,255,0.06)'
                }},
                legend: {{
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: 1.02,
                    xanchor: 'right',
                    x: 1
                }}
            }};

            Plotly.newPlot('top-bar-div', traces, layout);
        }}

        // Render Landscape Plot
        function updateLandscapePlot() {{
            const datasetKey = document.getElementById("landscape-select").value;
            const plotType = document.getElementById("plot-type-select").value;
            
            let zData;
            if (datasetKey === 'exact') zData = data.zExact;
            else if (datasetKey === 'ensemble') zData = data.zEns;
            else zData = data.zBase;

            const targetDiv = 'landscape-div';
            
            if (plotType === 'surface') {{
                const trace = {{
                    z: zData,
                    x: data.xCoords,
                    y: data.yCoords,
                    type: 'surface',
                    colorscale: 'Plasma',
                    colorbar: {{title: 'Probability', tickfont: {{color: '#fff'}}}}
                }};
                
                const layout = {{
                    ...commonLayout,
                    scene: {{
                        xaxis: {{title: `Lower ${{data.n1}} Qubits (binary)`, tickfont: {{size: 9}}, backgroundcolor: '#120f26', gridcolor: 'rgba(255,255,255,0.05)'}},
                        yaxis: {{title: `Upper ${{data.n2}} Qubits (binary)`, tickfont: {{size: 9}}, backgroundcolor: '#120f26', gridcolor: 'rgba(255,255,255,0.05)'}},
                        zaxis: {{title: 'Probability', backgroundcolor: '#120f26', gridcolor: 'rgba(255,255,255,0.05)'}},
                        camera: {{eye: {{x: 1.5, y: 1.5, z: 1.2}}}}
                    }},
                    margin: {{t: 0, r: 0, b: 0, l: 0}}
                }};
                
                Plotly.newPlot(targetDiv, [trace], layout);
            }} else {{
                // 2D Heatmap
                const trace = {{
                    z: zData,
                    x: data.xCoords,
                    y: data.yCoords,
                    type: 'heatmap',
                    colorscale: 'Plasma',
                    colorbar: {{title: 'Probability'}}
                }};
                
                const layout = {{
                    ...commonLayout,
                    xaxis: {{title: `Lower ${{data.n1}} Qubits (binary space)`, font: {{family: 'JetBrains Mono, monospace'}}, tickangle: 45}},
                    yaxis: {{title: `Upper ${{data.n2}} Qubits (binary space)`, font: {{family: 'JetBrains Mono, monospace'}}}},
                }};
                
                Plotly.newPlot(targetDiv, [trace], layout);
            }}
        }}

        // Render Correlation Heatmap
        function drawCorrelationHeatmap() {{
            const matrixKey = document.getElementById("correlation-select").value;
            let mat;
            let title;
            
            if (matrixKey === 'J') {{
                mat = data.J;
                title = 'Hebbian Coupling Matrix J';
            }} else if (matrixKey === 'exact') {{
                mat = data.C_exact;
                title = 'Exact Boltzmann Correlations';
            }} else if (matrixKey === 'ensemble') {{
                mat = data.C_ens;
                title = 'Ensemble Sampled Correlations';
            }} else {{
                mat = data.C_base;
                title = 'Baseline Sampled Correlations';
            }}

            if (!mat || mat.length === 0) {{
                document.getElementById('correlation-div').innerHTML = '<div style="color:rgba(255,255,255,0.4); text-align:center; padding:100px;">Data unavailable for this selection.</div>';
                return;
            }}

            const indices = Array.from({{length: data.nQubits}}, (_, i) => `Spin ${{i}}`);

            const trace = {{
                z: mat,
                x: indices,
                y: indices,
                type: 'heatmap',
                colorscale: 'RdBu',
                zmin: -1.0,
                zmax: 1.0,
                colorbar: {{title: 'Correlation value', tickfont: {{color: '#fff'}}}}
            }};

            const layout = {{
                ...commonLayout,
                xaxis: {{title: 'Spin Index i', side: 'bottom'}},
                yaxis: {{title: 'Spin Index j', autorange: 'reversed'}},
                margin: {{t: 30, r: 30, b: 60, l: 60}}
            }};

            Plotly.newPlot('correlation-div', [trace], layout);
        }}

        // Initial draws
        drawTopBarChart();
        updateLandscapePlot();
        drawCorrelationHeatmap();

        // Responsive adjustment
        window.onresize = function() {{
            Plotly.Plots.resize(document.getElementById('landscape-div'));
            Plotly.Plots.resize(document.getElementById('top-bar-div'));
            Plotly.Plots.resize(document.getElementById('correlation-div'));
        }};
    </script>
</body>
</html>
"""

    path = output_manager.get_path("hopfield_interactive.html")
    with open(path, 'w') as f:
        f.write(html_content)
    print(f"Saved premium interactive dashboard to: {path}")


def _save_raw_data(output_manager, exact_probs, baseline_probs, ensemble_probs, model_probs_list, weights, dataset, C_exact, C_base, C_ens, J):
    """Serialize all distributions, correlation matrices, weights, and parameters to a JSON file."""
    n_qubits = dataset.n_qubits
    patterns_list = dataset.patterns.tolist() if hasattr(dataset, 'patterns') else []
    weights_list = [float(w) for w in weights] if weights is not None else []
    
    data_payload = {
        'n_qubits': n_qubits,
        'exact_probs': exact_probs.tolist(),
        'baseline_probs': baseline_probs.tolist(),
        'ensemble_probs': ensemble_probs.tolist(),
        'model_probs': [p.tolist() for p in model_probs_list],
        'weights': weights_list,
        'patterns': patterns_list,
        'J': J.tolist(),
        'C_exact': C_exact.tolist(),
        'C_base': C_base.tolist() if C_base is not None else [],
        'C_ens': C_ens.tolist() if C_ens is not None else [],
        'beta': float(dataset.beta) if hasattr(dataset, 'beta') else 2.0
    }
    
    path = output_manager.get_path("hopfield_viz_data.json")
    with open(path, 'w') as f:
        json.dump(data_payload, f, indent=2)
    print(f"Saved raw visualization data to: {path}")


def _save_latex_table(output_manager, exact_probs, baseline_probs, ensemble_probs, n_qubits, top_k=10):
    """Generate a clean, professional LaTeX table comparing the top-K states."""
    top_k = min(top_k, 2**n_qubits)
    top_k_indices = np.argsort(exact_probs)[::-1][:top_k]
    
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison of top-K state probabilities for the Hopfield network (" + str(n_qubits) + " qubits).}",
        "\\label{tab:hopfield_top_states}",
        "\\begin{tabular}{lccccc}",
        "\\hline\\hline",
        "\\textbf{Bitstring} & \\textbf{Exact Prob} & \\textbf{Ensemble Prob} & \\textbf{Baseline Prob} & \\textbf{Ens. Error} & \\textbf{Base. Error} \\\\",
        "\\hline"
    ]
    
    for idx in top_k_indices:
        bitstr = format(idx, f'0{n_qubits}b')
        p_ex = exact_probs[idx]
        p_ens = ensemble_probs[idx]
        p_base = baseline_probs[idx]
        diff_ens = abs(p_ens - p_ex)
        diff_base = abs(p_base - p_ex)
        
        latex.append(
            f"\\texttt{{{bitstr}}} & {p_ex:.5f} & {p_ens:.5f} & {p_base:.5f} & {diff_ens:.5f} & {diff_base:.5f} \\\\"
        )
        
    latex.extend([
        "\\hline\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    latex_content = "\n".join(latex) + "\n"
    
    path = output_manager.get_path("hopfield_top_states.tex")
    with open(path, 'w') as f:
        f.write(latex_content)
    print(f"Saved LaTeX table to: {path}")
