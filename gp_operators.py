"""
gp_operators.py
---------------
The three genetic operators that drive evolution:

1. Tournament Selection -> pick good parents
2. Subtree Crossover    -> mix two parents to create a child
3. Subtree Mutation     -> randomly change part of a formula
"""

import random
from node import get_all_nodes, tree_depth
from generator import generate_tree


def tournament_selection(population, fitnesses, k=3):
    """Randomly pick k individuals, return a COPY of the best (lowest fitness) one."""
    idxs = random.sample(range(len(population)), k)
    best_idx = min(idxs, key=lambda i: fitnesses[i])
    return population[best_idx].copy()


def crossover(parent1, parent2):
    """
    Create one child by swapping a random subtree of parent1
    with a random subtree of parent2.

    Steps:
      1. Copy parent1 -> this becomes the offspring
      2. Copy parent2 -> this is the "donor"
      3. Pick a random node in the offspring, and a random node in the donor
      4. Replace the offspring node's content with the donor node's content
    """
    offspring = parent1.copy()
    donor = parent2.copy()

    offspring_nodes = get_all_nodes(offspring)
    donor_nodes = get_all_nodes(donor)

    target_node = random.choice(offspring_nodes)   # node to be replaced (inside offspring)
    source_node = random.choice(donor_nodes)         # subtree to copy in (from donor)

    target_node.type = source_node.type
    target_node.value = source_node.value
    target_node.children = source_node.children

    return offspring


def mutate(tree, variables, max_depth=3, mutation_rate=0.2):
    """With probability mutation_rate, replace a random subtree with a fresh random subtree."""
    tree = tree.copy()
    if random.random() < mutation_rate:
        nodes = get_all_nodes(tree)
        target_node = random.choice(nodes)
        new_subtree = generate_tree(random.randint(1, max_depth), variables,
                                     method=random.choice(['full', 'grow']))
        target_node.type = new_subtree.type
        target_node.value = new_subtree.value
        target_node.children = new_subtree.children
    return tree
