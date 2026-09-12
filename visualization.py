"""
visualization.py
-----------------
All Matplotlib plots required by the project. Every plot is SAVED as a
PNG file inside the 'plots' folder (so it works even when running from
a plain terminal, not just Jupyter) and also shown on screen if a
display is available.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from operators import evaluate_tree

PLOTS_DIR = "plots"


def _save_and_show(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f"  [saved plot -> {path}]")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


def plot_fitness_history(history, dataset_name="dataset", filename="fitness_vs_generation.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history, color='crimson', linewidth=2)
    ax.set_title(f"Best Fitness vs Generation ({dataset_name})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (lower = better)")
    ax.grid(alpha=0.3)
    _save_and_show(fig, filename)


def plot_actual_vs_predicted(y_true, y_pred, dataset_name="dataset", filename="actual_vs_predicted.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, color='teal', alpha=0.7)
    lims = [min(min(y_true), y_pred.min()), max(max(y_true), y_pred.max())]
    ax.plot(lims, lims, 'k--', label="Perfect prediction")
    ax.set_title(f"Actual vs Predicted ({dataset_name})")
    ax.set_xlabel("Actual y")
    ax.set_ylabel("Predicted y")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_and_show(fig, filename)


def plot_error_distribution(y_true, y_pred, dataset_name="dataset", filename="error_distribution.png"):
    errors = np.array(y_true) - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(errors, bins=15, color='orange', edgecolor='black')
    ax.set_title(f"Error Distribution ({dataset_name})")
    ax.set_xlabel("Error (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    _save_and_show(fig, filename)


def plot_complexity_vs_fitness(unique_formulas, dataset_name="dataset", filename="complexity_vs_fitness.png"):
    complexities = [v[3] for v in unique_formulas.values()]
    fits = [v[1] for v in unique_formulas.values()]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(complexities, fits, alpha=0.5, color='purple')
    ax.set_title(f"Complexity vs Fitness ({dataset_name})")
    ax.set_xlabel("Complexity (node count)")
    ax.set_ylabel("Fitness")
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    _save_and_show(fig, filename)


def plot_formula_curve(tree, X, y, var_name, formula_str, dataset_name="dataset",
                        filename="formula_curve.png"):
    """Only meaningful for single-variable datasets: shows the discovered curve vs actual points."""
    x_vals = np.array([row[var_name] for row in X])
    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
    y_line = np.array([evaluate_tree(tree, {var_name: v}) for v in x_line])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_vals, y, color='steelblue', label='Actual data', alpha=0.7)
    ax.plot(x_line, y_line, color='red', linewidth=2, label=f'Discovered: {formula_str}')
    ax.set_title(f"Discovered Formula Curve vs Actual Data ({dataset_name})")
    ax.set_xlabel(var_name)
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save_and_show(fig, filename)
