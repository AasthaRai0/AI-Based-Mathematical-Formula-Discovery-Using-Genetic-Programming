# AI-Based Mathematical Formula Discovery Using Genetic Programming

A complete **Genetic Programming (GP)** engine, built **100% from scratch in Python**
(no symbolic-regression library, no gradient descent), that automatically discovers
mathematical formulas from input–output data. This is called **Symbolic Regression**.

---

## 1. Setup (one time only)

```
pip install -r requirements.txt
```

This installs: `numpy`, `matplotlib`, `pandas`, `sympy`
(`sympy` is used ONLY to pretty-print/simplify the final formula for display —
it plays no role in the actual GP search algorithm, which is entirely hand-written).

---

## 2. How to run — there are two scripts

### ▶ `discover.py` — INTERACTIVE (use this for your live teacher demo)

```
python discover.py
```

You give it data (typed manually, or from a CSV file) and it prints the discovered
equation directly.

**Step-by-step:**
1. Choose how to provide data:
   - `1` → Type rows manually, e.g. `x, y` pairs
   - `2` → Load from a CSV file (you give the file path + which column is the target)
2. Press Enter to accept the default GP settings (population size / generations /
   max tree depth), or type your own numbers.
3. The program runs and prints the final result in a clearly boxed section:

```
################################################################
#                   FINAL DISCOVERED FORMULA                   #
################################################################

    y = 2*x + 0.9989

    Accuracy   : R^2 = 1.000000
    Error      : MSE = 0.000001   |   MAE = 0.001133
    Complexity : 6 nodes

################################################################
(Raw GP tree form, before simplification: (x-(sin(4.76)-x)))
```

followed by a Top-5 leaderboard (simplified formulas) and an actual-vs-predicted table.

**Example manual session** (you type: 1,3 / 2,5 / 3,7 / 4,9 / 5,11 / 6,13 → hidden formula is y = 2x+1):

```
Choose 1 or 2: 1
Number of input variables [default=1]: 1
Enter variable names separated by commas [default=x]:
Row 1 > 1, 3
Row 2 > 2, 5
Row 3 > 3, 7
Row 4 > 4, 9
Row 5 > 5, 11
Row 6 > 6, 13
Row 7 > done
What should the output/target be called? [default=y]:
Population size [default=400]:
Number of generations [default=40]:
Maximum tree depth [default=5]:
```
→ Output: `y = 2*x + 0.9989` with R² = 1.000000 (essentially the exact hidden formula, 2x+1)

**Example CSV session** (using the provided `sample_data/ohms_law.csv`, hidden formula V = I×R):
```
Choose 1 or 2: 2
Enter the path to your CSV file: sample_data/ohms_law.csv
Columns: ['current', 'resistance', 'voltage']
Which column is the TARGET/output column? voltage
```
→ Output: `voltage = current*resistance` with R² = 1.000000 (exact)

> **Note on harder formulas:** simple formulas (linear, single multiplication) are found
> almost instantly with the defaults. Formulas involving squares/combinations (like
> `sample_data/kinetic_energy.csv`, hidden formula KE = ½mv²) need a bigger search —
> try Population size = `800`, Generations = `80`, Max depth = `6` for those.

### ▶ `main.py` — AUTOMATED (runs on the 4 built-in sample datasets)

```
python main.py
```

Runs the same GP engine automatically on 4 pre-made datasets (linear, quadratic,
cubic, and a 2-variable formula), prints training progress + a Top-10 leaderboard +
MSE/MAE/R² for each, and **saves every required plot as a PNG** inside a newly
created `plots/` folder:
- Fitness vs Generation
- Actual vs Predicted
- Error Distribution
- Complexity vs Fitness
- Discovered Formula Curve vs Actual Data (single-variable datasets only)

Total runtime: roughly 1–2 minutes on a normal laptop.

---

## 3. File structure

| File | What it contains |
|---|---|
| `node.py` | The `Node` class — one node of an expression tree (copy, depth, node count, to-string) |
| `operators.py` | Safe/protected math (`+ - * / ^ sin cos sqrt log exp abs`) + tree evaluation |
| `generator.py` | Random expression tree generator + population initialization |
| `fitness.py` | Fitness function = MSE + complexity penalty |
| `gp_operators.py` | Tournament selection, subtree crossover, subtree mutation |
| `genetic_programming.py` | The main GP loop that ties everything together (the full pipeline) |
| `datasets.py` | The 4 sample datasets + a generic CSV loader for real data |
| `evaluation.py` | MSE / MAE / R² metrics + Top-K table printer |
| `visualization.py` | All Matplotlib plots (saved as PNG files) |
| `pretty_print.py` | Converts a raw GP tree into a clean, simplified, human-readable formula (via SymPy, display-only) |
| `main.py` | Automated demo — runs GP on all 4 built-in sample datasets + saves plots |
| `discover.py` | **Interactive demo — run this for the live teacher demo.** Type/upload your own data, get the discovered equation. |
| `sample_data/ohms_law.csv` | Sample CSV — hidden formula: `voltage = current * resistance` (Ohm's Law) |
| `sample_data/kinetic_energy.csv` | Sample CSV — hidden formula: `kinetic_energy = 0.5 * mass * velocity^2` |
| `requirements.txt` | Python dependencies |

---

## 4. Using your own CSV dataset

**Easiest way:** run `python discover.py`, choose option `2`, then give the CSV path
and the name of the target/output column. All other columns are automatically
treated as input variables.

To use a CSV inside `main.py` instead, replace one of the `make_dataset_X()` calls with:

```python
from datasets import load_dataset_from_csv
X, y, variables = load_dataset_from_csv("your_file.csv", target_column="your_target_column_name")
```

---

## 5. Key project rules followed

- ❌ No Gradient Descent used anywhere.
- ❌ No pre-built symbolic-regression/GP library (`gplearn`, `DEAP`, `PySR`, etc.) used as the algorithm.
- ✅ Genetic Programming implemented completely from scratch (tree, generator, fitness,
  selection, crossover, mutation, elitism, bloat control — all hand-written).
- ✅ Thousands of unique, non-hardcoded expressions generated and evolved per run.
- ✅ Invalid expressions (divide-by-zero, log/sqrt of negative numbers, overflow, NaN,
  Infinity) are handled safely and never crash the program.
- ✅ SymPy is used only to **display** a simplified version of the final formula —
  the search/discovery algorithm itself never calls it.

---