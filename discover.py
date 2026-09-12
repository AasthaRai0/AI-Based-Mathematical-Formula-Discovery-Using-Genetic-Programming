"""
discover.py
-----------
INTERACTIVE FORMULA DISCOVERY
==============================
This is the script your teacher will actually use to test the project
LIVE: the user types in their own data, and the program outputs the
discovered mathematical formula.

Run it with:

    python discover.py

You will be asked to either:
  1. Type data manually (row by row), or
  2. Load data from a CSV file

Then the from-scratch Genetic Programming engine runs and the discovered
equation is printed as the final output.
"""

import random
import numpy as np

from genetic_programming import genetic_programming, get_ranked_formulas
from evaluation import predict_all, compute_mae, compute_r2, print_topk_table
from datasets import load_dataset_from_csv
from pretty_print import prettify_tree


def ask_int(prompt, default):
    raw = input(f"{prompt} [default={default}]: ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print("  Invalid number, using default.")
        return default


def get_data_manually():
    print("\nHow many input variables does your data have?")
    print("  (e.g. type 1 if you only have x -> y)")
    print("  (e.g. type 2 if you have x, y -> output)")
    n_vars = ask_int("Number of input variables", 1)

    default_names = ['x', 'y', 'z', 'w'][:n_vars]
    raw_names = input(f"Enter variable names separated by commas [default={','.join(default_names)}]: ").strip()
    if raw_names == "":
        var_names = default_names
    else:
        var_names = [v.strip() for v in raw_names.split(',')]

    print(f"\nNow enter your data rows.")
    print(f"Each row must have {n_vars + 1} numbers separated by commas:")
    print(f"    {', '.join(var_names)}, target_output")
    print("Example row:", ", ".join(["2.5"] * n_vars) + ", 6.0")
    print("Type 'done' (without quotes) when you have entered all rows.\n")

    X, y = [], []
    row_num = 1
    while True:
        line = input(f"Row {row_num} > ").strip()
        if line.lower() == "done":
            break
        if line == "":
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != n_vars + 1:
            print(f"  Expected {n_vars + 1} numbers ({n_vars} inputs + 1 target), got {len(parts)}. Try again.")
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            print("  Could not read those as numbers. Try again.")
            continue
        row_dict = {var_names[i]: values[i] for i in range(n_vars)}
        X.append(row_dict)
        y.append(values[-1])
        row_num += 1

    if len(X) < 4:
        print("\n  WARNING: You entered very few rows. GP works much better with 15+ rows.")

    target_name = input("\nWhat should the output/target be called? [default=y]: ").strip()
    if target_name == "":
        target_name = "y"

    return X, y, var_names, target_name


def get_data_from_csv():
    path = input("\nEnter the path to your CSV file: ").strip()
    print("Column names found in the file will be shown below.")
    import pandas as pd
    preview = pd.read_csv(path)
    print("Columns:", list(preview.columns))
    target_column = input("Which column is the TARGET/output column? ").strip()
    X, y, variables = load_dataset_from_csv(path, target_column)
    print(f"\nLoaded {len(X)} rows. Input variables detected: {variables}")
    return X, y, variables, target_column


def main():
    print(r"""
    ################################################################
    #     AI-BASED MATHEMATICAL FORMULA DISCOVERY (Live Demo)      #
    #     Powered by Genetic Programming - built from scratch      #
    ################################################################
    """)

    print("How would you like to provide your data?")
    print("  1. Type data manually")
    print("  2. Load data from a CSV file")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "2":
        X, y, variables, target_name = get_data_from_csv()
    else:
        X, y, variables, target_name = get_data_manually()

    if len(X) == 0:
        print("\nNo data was entered. Exiting.")
        return

    print("\n--- GP Settings (press Enter to accept defaults) ---")
    pop_size = ask_int("Population size", 400)
    generations = ask_int("Number of generations", 40)
    max_depth = ask_int("Maximum tree depth", 5)

    # Fix the seed AFTER reading settings so re-runs with same data are reproducible
    random.seed(42)
    np.random.seed(42)

    print("\nRunning Genetic Programming... this may take a moment.\n")
    unique_formulas, history = genetic_programming(
        X, y, variables,
        pop_size=pop_size, generations=generations, max_depth=max_depth,
        elitism=6, lam=0.01, tournament_k=3, mutation_rate=0.25,
        verbose=True
    )

    ranked = get_ranked_formulas(unique_formulas)
    best_str, (best_tree, best_fit, best_mse, best_comp) = ranked[0]

    predictions = predict_all(best_tree, X)
    mae = compute_mae(y, predictions)
    r2 = compute_r2(y, predictions)

    # Convert the raw GP tree-formula into a clean, human-readable equation
    simplified = prettify_tree(best_tree)
    if simplified is None:
        simplified = best_str + "   (already in simplest form GP could find)"

    # ---------------------------------------------------------------
    # THE MAIN ANSWER - printed big, clear, and by itself
    # ---------------------------------------------------------------
    box_width = 64
    print("\n" + "#" * box_width)
    print("#" + "FINAL DISCOVERED FORMULA".center(box_width - 2) + "#")
    print("#" * box_width)
    print(f"\n    {target_name} = {simplified}\n")
    print(f"    Accuracy   : R^2 = {r2:.6f}")
    print(f"    Error      : MSE = {best_mse:.6f}   |   MAE = {mae:.6f}")
    print(f"    Complexity : {best_comp} nodes")
    print("\n" + "#" * box_width)
    print(f"(Raw GP tree form, before simplification: {best_str})")

    # ---------------------------------------------------------------
    # Supporting detail below the main answer
    # ---------------------------------------------------------------
    print(f"\nTotal unique formulas explored during evolution: {len(unique_formulas)}")

    print("\n--- TOP-5 FORMULAS DISCOVERED (simplified) ---")
    print(f"{'Rank':<5}{'Formula':<35}{'MSE':<15}{'Complexity':<10}")
    print("-" * 65)
    for i, (formula, (tree, fit, mse, comp)) in enumerate(ranked[:5]):
        clean = prettify_tree(tree)
        if clean is None:
            clean = formula
        clean_disp = clean if len(clean) <= 32 else clean[:29] + "..."
        print(f"{i + 1:<5}{clean_disp:<35}{mse:<15.6f}{comp:<10}")

    print("\n--- ACTUAL vs PREDICTED (first 10 rows) ---")
    print(f"{'Input':<40}{'Actual':<15}{'Predicted':<15}")
    print("-" * 70)
    for i in range(min(10, len(y))):
        input_str = ", ".join(f"{k}={v:.2f}" for k, v in X[i].items())
        print(f"{input_str:<40}{y[i]:<15.3f}{predictions[i]:<15.3f}")


if __name__ == "__main__":
    main()