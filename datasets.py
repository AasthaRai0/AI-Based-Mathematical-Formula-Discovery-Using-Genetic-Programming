"""
datasets.py
-----------
Sample datasets for testing the GP engine, plus a generic CSV loader
for real-world / user-provided data.

The GP algorithm is NEVER told these formulas - it only sees the
numeric X (inputs) and y (output) values.
"""

import numpy as np


def make_dataset_1(n=40, noise=0.0):
    """y = 2x + 1"""
    xs = np.linspace(-10, 10, n)
    ys = 2 * xs + 1 + np.random.normal(0, noise, n)
    X = [{'x': v} for v in xs]
    y = list(ys)
    return X, y, ['x']


def make_dataset_2(n=40, noise=0.0):
    """y = x^2"""
    xs = np.linspace(-10, 10, n)
    ys = xs ** 2 + np.random.normal(0, noise, n)
    X = [{'x': v} for v in xs]
    y = list(ys)
    return X, y, ['x']


def make_dataset_3(n=40, noise=0.0):
    """y = 3x^2 + 2x + 1"""
    xs = np.linspace(-10, 10, n)
    ys = 3 * xs ** 2 + 2 * xs + 1 + np.random.normal(0, noise, n)
    X = [{'x': v} for v in xs]
    y = list(ys)
    return X, y, ['x']


def make_dataset_4(n=60, noise=0.0):
    """output = 2x + y + 1  (two input variables x, y)"""
    xs = np.random.uniform(-10, 10, n)
    ys = np.random.uniform(-10, 10, n)
    out = 2 * xs + ys + 1 + np.random.normal(0, noise, n)
    X = [{'x': a, 'y': b} for a, b in zip(xs, ys)]
    y = list(out)
    return X, y, ['x', 'y']


def load_dataset_from_csv(path, target_column):
    """
    Generic loader for a user-provided CSV file.
    All columns except `target_column` are treated as input variables.
    """
    import pandas as pd
    df = pd.read_csv(path)
    variables = [c for c in df.columns if c != target_column]
    X = [{v: row[v] for v in variables} for _, row in df.iterrows()]
    y = list(df[target_column].values)
    return X, y, variables
