"""
node.py
--------
This file defines the Node class, which represents ONE node of a
mathematical expression tree.

Example: the formula  (2*x) + 1  is stored as this tree:

           +
          / \\
         *   1
        / \\
       2   x

Node types used in this project:
    'var'   -> a variable, e.g. x, y
    'const' -> a random numeric constant, e.g. 2, -3.5
    'op'    -> a binary operator (+ - * / ^) with 2 children
    'func'  -> a unary function (sin cos sqrt log exp abs) with 1 child
"""


class Node:
    """A single node of an expression tree."""

    def __init__(self, node_type, value=None, children=None):
        self.type = node_type                       # 'var', 'const', 'op', or 'func'
        self.value = value                           # variable name / number / operator symbol / function name
        self.children = children if children is not None else []

    def copy(self):
        """
        Deep copy of this tree.
        This is VERY important - crossover and mutation always work on
        COPIES of parents, never on the original trees, otherwise the
        population would get silently corrupted.
        """
        return Node(self.type, self.value, [child.copy() for child in self.children])


def count_nodes(node):
    """Total number of nodes in the tree -> used as the 'complexity' of a formula."""
    return 1 + sum(count_nodes(child) for child in node.children)


def tree_depth(node):
    """Depth (height) of the tree."""
    if not node.children:
        return 1
    return 1 + max(tree_depth(child) for child in node.children)


def get_all_nodes(node, collected=None):
    """Flatten the tree into a list of all its nodes (used by crossover & mutation)."""
    if collected is None:
        collected = []
    collected.append(node)
    for child in node.children:
        get_all_nodes(child, collected)
    return collected


def tree_to_string(node):
    """Convert a tree back into a readable infix mathematical string, e.g. '((2*x)+1)'."""
    if node.type == 'var':
        return str(node.value)
    if node.type == 'const':
        return str(node.value)
    if node.type == 'op':
        left = tree_to_string(node.children[0])
        right = tree_to_string(node.children[1])
        return f"({left}{node.value}{right})"
    if node.type == 'func':
        return f"{node.value}({tree_to_string(node.children[0])})"


if __name__ == "__main__":
    # Quick manual test: build (2*x) + 1 by hand and print it
    demo_tree = Node('op', '+', [
        Node('op', '*', [Node('const', 2), Node('var', 'x')]),
        Node('const', 1)
    ])
    print("Manual tree ->", tree_to_string(demo_tree))
    print("Node count   ->", count_nodes(demo_tree))
    print("Tree depth   ->", tree_depth(demo_tree))
