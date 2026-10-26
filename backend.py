"""
backend.py
----------
A thin wrapper between the Streamlit UI (app.py) and the Genetic Programming
engine. The UI only calls discover_formula(); everything else (cleaning the
data, running GP, simplifying the formula, computing metrics) happens here.
"""

import re
import random
import numpy as np
import pandas as pd

from genetic_programming import genetic_programming, get_ranked_formulas
from evaluation import predict_all, compute_mae, compute_r2
from pretty_print import prettify_tree, tree_to_sympy
from node import tree_to_string

# Search effort presets: (population size, generations, max tree depth)
EFFORT_PRESETS = {
    "Quick":    (300, 30, 5),
    "Standard": (600, 60, 6),
    "Thorough": (1000, 100, 6),
}

MAX_ROWS_FOR_SEARCH = 300   # GP search uses at most this many rows (keeps it fast)


def _clean_name(name):
    """Make a column name safe to use as a variable name in a formula."""
    cleaned = re.sub(r"\W+", "_", str(name).strip()).strip("_")
    if cleaned == "" or cleaned[0].isdigit():
        cleaned = "v_" + cleaned
    return cleaned


def prepare_data(df, target_column):
    """
    Turn a pandas DataFrame into the (X, y, variables) format the GP engine
    expects. Returns (X, y, variables, target_name, ignored_columns, n_dropped).
    Raises ValueError with a friendly message if the data is unusable.
    """
    df = df.copy()
    df.columns = [_clean_name(c) for c in df.columns]
    target_name = _clean_name(target_column)

    if target_name not in df.columns:
        raise ValueError("Target column not found in the file.")

    numeric_cols = list(df.select_dtypes(include="number").columns)
    if target_name not in numeric_cols:
        raise ValueError("The target column must contain numbers only.")

    variables = [c for c in numeric_cols if c != target_name]
    ignored = [c for c in df.columns if c not in numeric_cols]
    if len(variables) == 0:
        raise ValueError("Need at least one numeric input column besides the target.")

    before = len(df)
    df = df[variables + [target_name]].replace([np.inf, -np.inf], np.nan).dropna()
    n_dropped = before - len(df)

    if len(df) < 5:
        raise ValueError("Need at least 5 valid rows of data (found %d)." % len(df))

    return df, variables, target_name, ignored, n_dropped


def discover_formula(df, target_column, effort="Standard", seed=42):
    """
    Full pipeline: clean data -> run GP -> simplify best formula -> metrics.
    Returns a dict with everything the UI needs to display.
    """
    df, variables, target_name, ignored, n_dropped = prepare_data(df, target_column)

    # Use a sample of rows for the (slow, pure-Python) search, but report
    # final metrics on ALL rows.
    if len(df) > MAX_ROWS_FOR_SEARCH:
        search_df = df.sample(MAX_ROWS_FOR_SEARCH, random_state=seed)
    else:
        search_df = df

    X_search = search_df[variables].to_dict(orient="records")
    y_search = list(search_df[target_name].values)

    X_all = df[variables].to_dict(orient="records")
    y_all = list(df[target_name].values)

    pop_size, generations, max_depth = EFFORT_PRESETS[effort]

    random.seed(seed)
    np.random.seed(seed)

    unique_formulas, history = genetic_programming(
        X_search, y_search, variables,
        pop_size=pop_size, generations=generations, max_depth=max_depth,
        elitism=6, lam=0.01, tournament_k=3, mutation_rate=0.25,
        verbose=False
    )

    ranked = get_ranked_formulas(unique_formulas)
    best_str, (best_tree, best_fit, best_mse, best_comp) = ranked[0]

    predictions = predict_all(best_tree, X_all)
    mse = float(np.mean((np.array(y_all) - predictions) ** 2))
    mae = compute_mae(y_all, predictions)
    r2 = compute_r2(y_all, predictions)

    pretty = prettify_tree(best_tree)
    formula_text = pretty if pretty is not None else best_str

    return {
        "target": target_name,
        "variables": variables,
        "formula": formula_text,
        "raw_formula": best_str,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "complexity": best_comp,
        "n_rows": len(df),
        "ignored_columns": ignored,
        "n_dropped": n_dropped,
        "n_explored": len(unique_formulas),
    }
