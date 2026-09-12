"""
evaluation.py
-------------
Standard regression metrics, computed manually (no sklearn needed).
"""

import numpy as np
from operators import evaluate_tree


def predict_all(tree, X):
    """Run the formula (tree) on every row of X, return a numpy array of predictions."""
    return np.array([evaluate_tree(tree, row) for row in X])


def compute_mae(y_true, y_pred):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(y_true) - y_pred)))


def compute_r2(y_true, y_pred):
    """R^2 score: 1.0 = perfect fit, 0.0 = as good as predicting the mean."""
    y_true = np.array(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1 - ss_res / ss_tot)


def print_topk_table(ranked_formulas, k=10):
    """Print a Rank | Formula | MSE | Complexity table."""
    print(f"{'Rank':<5}{'Formula':<45}{'MSE':<15}{'Complexity':<10}")
    print("-" * 75)
    for i, (formula, (tree, fit, mse, comp)) in enumerate(ranked_formulas[:k]):
        formula_display = formula if len(formula) <= 42 else formula[:39] + "..."
        print(f"{i + 1:<5}{formula_display:<45}{mse:<15.6f}{comp:<10}")
