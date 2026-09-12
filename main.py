"""
main.py
-------
AI-BASED MATHEMATICAL FORMULA DISCOVERY USING GENETIC PROGRAMMING
====================================================================
This is the entry point of the project. Run it directly:

    python main.py

It will:
  1. Create 4 sample datasets (with hidden formulas the GP never sees)
  2. Run the from-scratch Genetic Programming engine on each of them
  3. Print the training progress, best formula, and a Top-K leaderboard
  4. Print evaluation metrics (MSE, MAE, R^2)
  5. Save all required plots as PNG files inside the 'plots/' folder

No symbolic-regression library and no gradient descent is used anywhere -
the entire GP algorithm (node.py, generator.py, fitness.py, gp_operators.py,
genetic_programming.py) is written from scratch.
"""

import random
import numpy as np

from datasets import make_dataset_1, make_dataset_2, make_dataset_3, make_dataset_4
from genetic_programming import genetic_programming, get_ranked_formulas
from evaluation import predict_all, compute_mae, compute_r2, print_topk_table
from visualization import (
    plot_fitness_history, plot_actual_vs_predicted,
    plot_error_distribution, plot_complexity_vs_fitness, plot_formula_curve
)

# Fix the random seed so results are reproducible during a live demo
random.seed(42)
np.random.seed(42)

# GP hyperparameters (feel free to tweak these for your demo)
GP_PARAMS = dict(
    pop_size=400,
    generations=40,
    max_depth=5,
    elitism=6,
    lam=0.01,
    tournament_k=3,
    mutation_rate=0.25,
)


def run_on_dataset(name, X, y, variables, single_var_plot=True):
    print("\n" + "=" * 78)
    print(f" RUNNING GENETIC PROGRAMMING ON: {name}")
    print("=" * 78)

    unique_formulas, history = genetic_programming(X, y, variables, verbose=True, **GP_PARAMS)

    ranked = get_ranked_formulas(unique_formulas)
    best_str, (best_tree, best_fit, best_mse, best_comp) = ranked[0]

    predictions = predict_all(best_tree, X)
    mae = compute_mae(y, predictions)
    r2 = compute_r2(y, predictions)

    print(f"\nTotal unique expressions discovered : {len(unique_formulas)}")
    print("\n--- BEST DISCOVERED FORMULA ---")
    print(f"Formula     : {best_str}")
    print(f"MSE         : {best_mse:.6f}")
    print(f"MAE         : {mae:.6f}")
    print(f"R^2 Score   : {r2:.6f}")
    print(f"Complexity  : {best_comp} nodes")

    print("\n--- TOP-10 UNIQUE FORMULAS ---")
    print_topk_table(ranked, k=10)

    print("\n--- SAMPLE ACTUAL vs PREDICTED ---")
    for i in range(min(5, len(y))):
        print(f"  actual = {y[i]:10.3f}   predicted = {predictions[i]:10.3f}")

    # ---- Visualizations ----
    safe_name = name.split(":")[0].strip().replace(" ", "_").lower()
    print("\n--- SAVING VISUALIZATIONS ---")
    plot_fitness_history(history, dataset_name=name, filename=f"{safe_name}_fitness_vs_generation.png")
    plot_actual_vs_predicted(y, predictions, dataset_name=name, filename=f"{safe_name}_actual_vs_predicted.png")
    plot_error_distribution(y, predictions, dataset_name=name, filename=f"{safe_name}_error_distribution.png")
    plot_complexity_vs_fitness(unique_formulas, dataset_name=name, filename=f"{safe_name}_complexity_vs_fitness.png")

    if single_var_plot and len(variables) == 1:
        plot_formula_curve(best_tree, X, y, variables[0], best_str,
                            dataset_name=name, filename=f"{safe_name}_formula_curve.png")

    return {
        "ranked": ranked, "history": history, "best_str": best_str,
        "best_tree": best_tree, "best_mse": best_mse, "mae": mae, "r2": r2
    }


def main():
    print(r"""
    ###########################################################
    #   AI-BASED MATHEMATICAL FORMULA DISCOVERY USING          #
    #   GENETIC PROGRAMMING  (Symbolic Regression, from scratch)#
    ###########################################################
    """)

    all_results = {}

    # Dataset 1: y = 2x + 1
    X1, y1, vars1 = make_dataset_1(n=40, noise=0.0)
    all_results["Dataset 1: y = 2x + 1"] = run_on_dataset("Dataset 1: y = 2x + 1", X1, y1, vars1)

    # Dataset 2: y = x^2
    X2, y2, vars2 = make_dataset_2(n=40, noise=0.0)
    all_results["Dataset 2: y = x^2"] = run_on_dataset("Dataset 2: y = x^2", X2, y2, vars2)

    # Dataset 3: y = 3x^2 + 2x + 1
    X3, y3, vars3 = make_dataset_3(n=40, noise=0.0)
    all_results["Dataset 3: y = 3x^2 + 2x + 1"] = run_on_dataset("Dataset 3: y = 3x^2 + 2x + 1", X3, y3, vars3)

    # Dataset 4: output = 2x + y + 1  (multi-variable, no curve plot possible)
    X4, y4, vars4 = make_dataset_4(n=60, noise=0.0)
    all_results["Dataset 4: output = 2x + y + 1"] = run_on_dataset(
        "Dataset 4: output = 2x + y + 1", X4, y4, vars4, single_var_plot=False)

    # ---- Final summary table across all datasets ----
    print("\n" + "=" * 78)
    print(" FINAL SUMMARY - ALL DATASETS")
    print("=" * 78)
    print(f"{'Dataset':<35}{'Best Formula':<25}{'MSE':<12}{'R^2':<10}")
    print("-" * 82)
    for name, res in all_results.items():
        formula_disp = res["best_str"] if len(res["best_str"]) <= 22 else res["best_str"][:19] + "..."
        print(f"{name:<35}{formula_disp:<25}{res['best_mse']:<12.6f}{res['r2']:<10.6f}")

    print("\nAll plots have been saved inside the 'plots/' folder.")
    print("Done.")


if __name__ == "__main__":
    main()
