"""
genetic_programming.py
-----------------------
The complete Genetic Programming pipeline for Symbolic Regression.

    Initial Population
          |
    Fitness Evaluation
          |
      Selection
          |
      Crossover
          |
      Mutation
          |
    New Population  (best individuals kept unchanged = Elitism)
          |
    Repeat for many generations
          |
    Best Mathematical Formula
"""

from node import tree_to_string, count_nodes, tree_depth
from generator import initialize_population
from fitness import compute_fitness, compute_mse
from gp_operators import tournament_selection, crossover, mutate
import random


def genetic_programming(X, y, variables, pop_size=400, generations=40,
                         max_depth=5, elitism=6, lam=0.01,
                         tournament_k=3, mutation_rate=0.25, crossover_rate=0.8,
                         verbose=True):
    """
    Runs the full Genetic Programming pipeline for symbolic regression.

    Returns:
        unique_formulas      : dict {formula_string: (tree, fitness, mse, complexity)}
        best_fitness_per_gen : list of best fitness value at each generation
    """
    population = initialize_population(pop_size, variables, max_depth)
    best_fitness_per_gen = []
    unique_formulas = {}

    for gen in range(generations):
        # ---- (a) Evaluate fitness of the whole population ----
        fitnesses = [compute_fitness(t, X, y, lam) for t in population]

        # ---- (b) Record every unique formula seen so far (duplicate removal) ----
        for t, f in zip(population, fitnesses):
            key = tree_to_string(t)
            if key not in unique_formulas or f < unique_formulas[key][1]:
                mse = compute_mse(t, X, y)
                comp = count_nodes(t)
                unique_formulas[key] = (t.copy(), f, mse, comp)

        order = sorted(range(len(population)), key=lambda i: fitnesses[i])
        best_fitness_per_gen.append(fitnesses[order[0]])

        if verbose and (gen % 5 == 0 or gen == generations - 1):
            print(f"  Gen {gen:3d} | Best fitness = {fitnesses[order[0]]:.6f} | "
                  f"Best formula = {tree_to_string(population[order[0]])}")

        # ---- (c) Elitism: carry the best individuals forward unchanged ----
        new_population = [population[i].copy() for i in order[:elitism]]

        # ---- (d) Fill the rest via Selection -> Crossover -> Mutation ----
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_k)
            parent2 = tournament_selection(population, fitnesses, tournament_k)

            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()

            child = mutate(child, variables, max_depth=3, mutation_rate=mutation_rate)

            # ---- (e) Bloat control: reject overly deep children ----
            if tree_depth(child) > max_depth + 2:
                child = parent1.copy()

            new_population.append(child)

        population = new_population

    return unique_formulas, best_fitness_per_gen


def get_ranked_formulas(unique_formulas):
    """Sort unique formulas dictionary by fitness (ascending = best first)."""
    return sorted(unique_formulas.items(), key=lambda kv: kv[1][1])
