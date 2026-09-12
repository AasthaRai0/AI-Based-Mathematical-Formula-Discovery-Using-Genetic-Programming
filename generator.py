"""
generator.py
------------
Randomly builds expression trees so the system explores MANY different
kinds of formulas (not a fixed few, and NEVER hardcoded).

Two classic GP methods are combined ("ramped half-and-half"):
    'full' -> every branch grows all the way to the max depth (bushy trees)
    'grow' -> branches can end early with a terminal, giving irregular shapes
"""

import random
from node import Node
from operators import OPERATORS_LIST, FUNCTIONS_LIST


def random_terminal(variables, const_range=(-5, 5)):
    """Return a random leaf node: either a variable or a random constant."""
    if random.random() < 0.5:
        return Node('var', value=random.choice(variables))
    else:
        return Node('const', value=round(random.uniform(*const_range), 2))


def generate_tree(depth, variables, method='grow'):
    """
    Recursively build a random expression tree.
    depth  -> how many more levels are allowed
    method -> 'full' (always grow to max depth) or 'grow' (can stop early)
    """
    # Stop condition: reached max depth, OR (in 'grow' mode) randomly decided to stop early
    if depth <= 0:
        return random_terminal(variables)
    if method == 'grow' and random.random() < 0.35:
        return random_terminal(variables)

    # Otherwise create an internal node: operator (75% chance) or function (25% chance)
    if random.random() < 0.75:
        op = random.choice(OPERATORS_LIST)
        left = generate_tree(depth - 1, variables, method)
        right = generate_tree(depth - 1, variables, method)
        return Node('op', value=op, children=[left, right])
    else:
        func = random.choice(FUNCTIONS_LIST)
        child = generate_tree(depth - 1, variables, method)
        return Node('func', value=func, children=[child])


def initialize_population(pop_size, variables, max_depth):
    """Create pop_size random trees with varied depths and generation methods."""
    population = []
    depths = list(range(2, max_depth + 1))
    for i in range(pop_size):
        depth = depths[i % len(depths)]
        method = 'full' if i % 2 == 0 else 'grow'
        population.append(generate_tree(depth, variables, method))
    return population


if __name__ == "__main__":
    from node import tree_to_string
    print("Sample random expressions (never hardcoded):")
    for _ in range(5):
        t = generate_tree(depth=3, variables=['x', 'y'], method=random.choice(['full', 'grow']))
        print(" ", tree_to_string(t))
