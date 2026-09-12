"""
fitness.py
----------
Fitness = MSE + lambda * complexity

- MSE (Mean Squared Error) rewards ACCURACY.
- complexity (number of nodes in the tree) is penalised by a small
  weight `lambda`, so the algorithm prefers the SIMPLEST formula that
  still fits well (Occam's Razor).
- LOWER fitness = BETTER formula.
- If a formula produces NaN / Infinity / huge error, it is instantly
  given a huge fitness value (1e12) so it is effectively eliminated,
  without ever crashing the program.
"""

import math
from operators import evaluate_tree
from node import count_nodes

BAD_FITNESS = 1e12


def compute_mse(tree, X, y):
    """Mean Squared Error of a tree's predictions against the true targets."""
    n = len(y)
    total_error = 0.0
    for i in range(n):
        pred = evaluate_tree(tree, X[i])
        if pred is None or math.isnan(pred) or math.isinf(pred):
            return BAD_FITNESS                     # invalid expression -> huge penalty
        total_error += (y[i] - pred) ** 2
        if total_error > BAD_FITNESS:               # early exit if error is exploding
            return BAD_FITNESS
    return total_error / n


def compute_fitness(tree, X, y, lam=0.01):
    """Fitness = MSE + lambda * complexity. Lower is better."""
    mse = compute_mse(tree, X, y)
    if mse >= BAD_FITNESS:
        return BAD_FITNESS
    complexity = count_nodes(tree)
    return mse + lam * complexity
