"""
backend.py
----------
A thin wrapper between the Streamlit UI (app.py) and the Genetic Programming
engine. The UI only calls discover_formula(); everything else (cleaning the
data, running GP, simplifying the formula, computing metrics) happens here.
"""

import re
import time
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


def validate_dataframe(df):
    """
    Inspect a raw DataFrame and report data-quality problems WITHOUT changing it.
    The UI shows this report; prepare_data() applies the actual cleaning.
    """
    numeric_cols = list(df.select_dtypes(include="number").columns)
    non_numeric = [c for c in df.columns if c not in numeric_cols]

    missing_by_col = {}
    for c in df.columns:
        n_missing = int(df[c].isna().sum())
        if n_missing > 0:
            missing_by_col[c] = n_missing

    inf_by_col, constant_cols = {}, []
    for c in numeric_cols:
        values = df[c].astype("float64").to_numpy()
        n_inf = int(np.isinf(values).sum())
        if n_inf > 0:
            inf_by_col[c] = n_inf
        if df[c].nunique(dropna=True) <= 1:
            constant_cols.append(c)

    cleaned = [_clean_name(c) for c in df.columns]
    name_clashes = sorted({c for c, k in zip(df.columns, cleaned) if cleaned.count(k) > 1})

    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "numeric_columns": numeric_cols,
        "non_numeric_columns": non_numeric,
        "missing_by_column": missing_by_col,
        "total_missing": int(sum(missing_by_col.values())),
        "inf_by_column": inf_by_col,
        "total_inf": int(sum(inf_by_col.values())),
        "constant_columns": constant_cols,
        "name_clashes": name_clashes,
    }


def count_unusable_rows(df, columns):
    """How many rows have a missing or infinite value in any of the given columns."""
    sub = df[list(columns)].apply(pd.to_numeric, errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan)
    return int(sub.isna().any(axis=1).sum())


def prepare_data(df, target_column, features=None):
    """
    Turn a pandas DataFrame into the (X, y, variables) format the GP engine
    expects. Returns (df, variables, target_name, ignored_columns, n_dropped).
    `features` (optional) = list of original column names to use as inputs;
    if None, every numeric column except the target is used.
    Raises ValueError with a friendly message if the data is unusable.
    """
    df = df.copy()
    cleaned = [_clean_name(c) for c in df.columns]
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Two columns have names that become identical after cleaning "
                         "(for example 'a b' and 'a_b'). Please rename one of them.")
    df.columns = cleaned
    target_name = _clean_name(target_column)

    if target_name not in df.columns:
        raise ValueError("Target column not found in the file.")

    numeric_cols = list(df.select_dtypes(include="number").columns)
    if target_name not in numeric_cols:
        raise ValueError("The target column must contain numbers only.")

    if features is None:
        variables = [c for c in numeric_cols if c != target_name]
    else:
        wanted = [_clean_name(f) for f in features]
        if target_name in wanted:
            raise ValueError("The target column cannot also be an input feature.")
        bad = [f for f in wanted if f not in numeric_cols]
        if bad:
            raise ValueError("These input columns are not numeric: " + ", ".join(bad))
        variables = [c for c in numeric_cols if c in wanted]
    ignored = [c for c in df.columns if c not in numeric_cols]
    if len(variables) == 0:
        raise ValueError("Select at least one numeric input column besides the target.")

    before = len(df)
    df = df[variables + [target_name]].replace([np.inf, -np.inf], np.nan).dropna()
    n_dropped = before - len(df)

    if len(df) < 5:
        raise ValueError("Need at least 5 valid rows of data (found %d)." % len(df))

    return df, variables, target_name, ignored, n_dropped


def _metrics(y_true, y_pred):
    """MSE / MAE / R2 for one set of predictions."""
    y_true = np.array(y_true, dtype=float)
    return {
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "mae": compute_mae(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
    }


def discover_formula(df, target_column, effort="Standard", seed=42,
                     features=None, test_size=0.0, progress_callback=None):
    """
    Full pipeline: clean data -> (optional train/test split) -> run GP ->
    simplify best formula -> metrics. Returns a dict with everything the UI needs.

    features          : optional list of input columns (default: all numeric columns)
    test_size         : fraction of rows held out for testing (0 = use all rows to train)
    progress_callback : optional function called after every GP generation
    """
    started = time.time()
    df, variables, target_name, ignored, n_dropped = prepare_data(df, target_column, features)

    # ---- optional hold-out test set (never seen by the GP search) ----
    n = len(df)
    if test_size and test_size > 0:
        n_test = max(2, int(round(n * test_size)))
        if n - n_test < 5:
            raise ValueError("Not enough rows for this test split. Lower the test size "
                             "or upload more rows.")
        perm = np.random.RandomState(seed).permutation(n)
        is_test = np.zeros(n, dtype=bool)
        is_test[perm[:n_test]] = True
        train_df, test_df = df[~is_test], df[is_test]
    else:
        train_df, test_df = df, None

    # Use a sample of training rows for the (slow, pure-Python) search, but report
    # final metrics on ALL training rows (and all test rows).
    if len(train_df) > MAX_ROWS_FOR_SEARCH:
        search_df = train_df.sample(MAX_ROWS_FOR_SEARCH, random_state=seed)
    else:
        search_df = train_df

    X_search = search_df[variables].to_dict(orient="records")
    y_search = list(search_df[target_name].values)

    pop_size, generations, max_depth = EFFORT_PRESETS[effort]

    random.seed(seed)
    np.random.seed(seed)

    unique_formulas, history = genetic_programming(
        X_search, y_search, variables,
        pop_size=pop_size, generations=generations, max_depth=max_depth,
        elitism=6, lam=0.01, tournament_k=3, mutation_rate=0.25,
        verbose=False, progress_callback=progress_callback
    )

    ranked = get_ranked_formulas(unique_formulas)
    best_str, (best_tree, best_fit, best_mse, best_comp) = ranked[0]

    # ---- predictions of the best formula ----
    X_train = train_df[variables].to_dict(orient="records")
    y_train = train_df[target_name].values
    pred_train = predict_all(best_tree, X_train)
    train_metrics = _metrics(y_train, pred_train)

    test_metrics, pred_test, y_test = None, None, None
    if test_df is not None:
        X_test = test_df[variables].to_dict(orient="records")
        y_test = test_df[target_name].values
        pred_test = predict_all(best_tree, X_test)
        test_metrics = _metrics(y_test, pred_test)

    X_all = df[variables].to_dict(orient="records")
    y_all = df[target_name].values
    pred_all = predict_all(best_tree, X_all)
    overall = _metrics(y_all, pred_all)

    pretty = prettify_tree(best_tree)
    formula_text = pretty if pretty is not None else best_str

    # ---- Top-5 distinct formulas (distinct after simplification) ----
    top_formulas, seen = [], set()
    for formula_str, (tree, fit, _search_mse, comp) in ranked[:25]:
        shown = prettify_tree(tree) or formula_str
        if shown in seen:
            continue
        seen.add(shown)
        entry = {
            "formula": shown,
            "complexity": comp,
            "fitness": float(fit),
            "train_mse": float(np.mean((y_train - predict_all(tree, X_train)) ** 2)),
            "test_mse": None,
        }
        if test_df is not None:
            entry["test_mse"] = float(np.mean((y_test - predict_all(tree, X_test)) ** 2))
        top_formulas.append(entry)
        if len(top_formulas) == 5:
            break

    # ---- predictions table (for CSV download) ----
    predictions_df = df[variables + [target_name]].copy()
    predictions_df.insert(0, "row_id", df.index)
    predictions_df = predictions_df.rename(columns={target_name: "actual_" + target_name})
    predictions_df["predicted_" + target_name] = pred_all
    predictions_df["error"] = y_all - pred_all
    predictions_df["split"] = "train"
    if test_df is not None:
        predictions_df.loc[test_df.index, "split"] = "test"

    return {
        "target": target_name,
        "variables": variables,
        "formula": formula_text,
        "raw_formula": best_str,
        "mse": overall["mse"],
        "mae": overall["mae"],
        "r2": overall["r2"],
        "complexity": best_comp,
        "n_rows": len(df),
        "ignored_columns": ignored,
        "n_dropped": n_dropped,
        "n_explored": len(unique_formulas),
        # ---- added for the improved UI ----
        "history": [float(h) for h in history],
        "top_formulas": top_formulas,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "predictions_df": predictions_df,
        "y_train": y_train, "pred_train": pred_train,
        "y_test": y_test, "pred_test": pred_test,
        "settings": {
            "effort": effort, "population": pop_size, "generations": generations,
            "max_depth": max_depth, "seed": seed, "test_size": test_size or 0.0,
            "n_train": len(train_df), "n_test": 0 if test_df is None else len(test_df),
            "n_search_rows": len(search_df),
        },
        "elapsed_seconds": time.time() - started,
    }