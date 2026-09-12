"""
operators.py
------------
Random expressions can easily contain things like 1/0, log(-4), sqrt(-9),
or numbers so large they overflow. Instead of letting Python crash, every
operator/function here is a "protected" (safe) version that ALWAYS returns
a normal float, never raises an exception.

If a formula is genuinely bad, it will simply get a very poor FITNESS score
later (see fitness.py) - the program itself never crashes.
"""

import math


def safe_add(a, b):
    return a + b


def safe_sub(a, b):
    return a - b


def safe_mul(a, b):
    return a * b


def safe_div(a, b):
    """Protected division: avoids division by (near) zero."""
    try:
        if abs(b) < 1e-6:
            return 1.0
        result = a / b
        if math.isnan(result) or math.isinf(result):
            return 1.0
        return result
    except Exception:
        return 1.0


def safe_pow(a, b):
    """Protected power: clips exponent, avoids complex numbers from negative bases."""
    try:
        b = max(-3, min(3, b))          # clip exponent so numbers don't explode
        base = abs(a)                    # avoid complex numbers from negative bases
        value = math.pow(base, b)
        if a < 0 and float(b).is_integer() and int(b) % 2 != 0:
            value = -value                # keep sign correct for odd integer powers
        if math.isnan(value) or math.isinf(value):
            return 1.0
        return value
    except Exception:
        return 1.0


def safe_sin(a):
    try:
        return math.sin(a)
    except Exception:
        return 0.0


def safe_cos(a):
    try:
        return math.cos(a)
    except Exception:
        return 0.0


def safe_sqrt(a):
    """Protected sqrt: uses abs() so negative numbers don't crash it."""
    try:
        return math.sqrt(abs(a))
    except Exception:
        return 0.0


def safe_log(a):
    """Protected log: uses abs() and avoids log(0)."""
    try:
        if abs(a) < 1e-6:
            return 0.0
        return math.log(abs(a))
    except Exception:
        return 0.0


def safe_exp(a):
    """Protected exp: clips input so it never overflows."""
    try:
        a = max(-20, min(20, a))
        return math.exp(a)
    except Exception:
        return 1.0


def safe_abs(a):
    try:
        return abs(a)
    except Exception:
        return 0.0


# Binary operators: symbol -> function(a, b)
OPERATORS = {
    '+': safe_add,
    '-': safe_sub,
    '*': safe_mul,
    '/': safe_div,
    '^': safe_pow,
}

# Unary functions: name -> function(a)
FUNCTIONS = {
    'sin': safe_sin,
    'cos': safe_cos,
    'sqrt': safe_sqrt,
    'log': safe_log,
    'exp': safe_exp,
    'abs': safe_abs,
}

OPERATORS_LIST = list(OPERATORS.keys())
FUNCTIONS_LIST = list(FUNCTIONS.keys())


def evaluate_tree(node, var_values):
    """
    Recursively evaluate a tree for one row of input data.
    var_values is a dict like {'x': 3.5, 'y': -2.0}
    """
    try:
        if node.type == 'var':
            return float(var_values[node.value])
        if node.type == 'const':
            return float(node.value)
        if node.type == 'op':
            args = [evaluate_tree(c, var_values) for c in node.children]
            return OPERATORS[node.value](*args)
        if node.type == 'func':
            arg = evaluate_tree(node.children[0], var_values)
            return FUNCTIONS[node.value](arg)
    except Exception:
        return 1e6          # anything unexpected -> safe large "bad" value
    return 1e6
