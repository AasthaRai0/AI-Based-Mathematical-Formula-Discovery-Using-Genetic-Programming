"""
pretty_print.py
----------------
The Genetic Programming engine produces formulas in a raw, fully-parenthesized
form, e.g.:

    ((x*(x+x))+((x*x)+((x/x)+(x+x))))

This is CORRECT, but hard for a human to read. This file uses SymPy
(a symbolic-math library) purely for DISPLAY/SIMPLIFICATION purposes -
it plays NO role in the actual Genetic Programming search/discovery
process, which remains 100% built from scratch.

prettify_formula() converts the raw string above into:

    3*x**2 + 2*x + 1

which is immediately recognisable as the correct answer.
"""

import sympy as sp


def round_floats(expr, digits=4):
    """Round every floating-point constant inside a sympy expression."""
    return expr.xreplace({n: round(float(n), digits) for n in expr.atoms(sp.Float)})


def tree_to_sympy(node):
    """
    Convert our own expression tree (Node object) directly into a SymPy
    expression - built to match the SAME "protected/safe" semantics used
    during evaluation (see operators.py), so the displayed formula never
    disagrees with what the GP engine actually computed.

    In particular: sqrt() and log() always take the absolute value of their
    argument first, exactly like safe_sqrt()/safe_log() do. This avoids
    SymPy introducing complex numbers (e.g. log of a negative constant)
    for formulas that are perfectly well-defined under our protected math.
    """
    if node.type == 'var':
        return sp.Symbol(node.value)
    if node.type == 'const':
        return sp.Float(float(node.value))
    if node.type == 'op':
        left = tree_to_sympy(node.children[0])
        right = tree_to_sympy(node.children[1])
        if node.value == '+':
            return left + right
        if node.value == '-':
            return left - right
        if node.value == '*':
            return left * right
        if node.value == '/':
            return left / right
        if node.value == '^':
            # Matches safe_pow(): uses the absolute value of the base
            return sp.Abs(left) ** right
    if node.type == 'func':
        arg = tree_to_sympy(node.children[0])
        if node.value == 'sin':
            return sp.sin(arg)
        if node.value == 'cos':
            return sp.cos(arg)
        if node.value == 'sqrt':
            return sp.sqrt(sp.Abs(arg))     # matches safe_sqrt()
        if node.value == 'log':
            return sp.log(sp.Abs(arg))      # matches safe_log()
        if node.value == 'exp':
            return sp.exp(arg)
        if node.value == 'abs':
            return sp.Abs(arg)
    raise ValueError(f"Unknown node type/value: {node.type}, {node.value}")


def prettify_tree(tree, digits=4):
    """
    Convert a GP expression tree into a clean, simplified, human-readable
    math string, e.g. '3*x**2 + 2*x + 1'.

    Returns None if simplification fails for any reason (caller should
    fall back to the raw tree_to_string() form in that case).
    """
    try:
        expr = tree_to_sympy(tree)
        simplified = sp.simplify(expr)
        simplified = round_floats(simplified, digits)
        result = str(simplified)
        # Guard against SymPy still producing something exotic (e.g. complex
        # numbers, oo) - if so, prefer the raw form instead.
        if 'I' in result.replace('x', '').replace('I' + '', 'I') and 'zoo' not in result:
            # crude but effective: check the free-symbol-stripped string for
            # a standalone imaginary unit token
            import re
            if re.search(r'(?<![a-zA-Z_])I(?![a-zA-Z_])', result):
                return None
        if 'zoo' in result or 'nan' in result:
            return None
        return result
    except Exception:
        return None
