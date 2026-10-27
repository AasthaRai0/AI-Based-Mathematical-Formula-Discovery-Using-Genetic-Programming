# """
# app.py — Streamlit UI for AI-Based Mathematical Formula Discovery
# Run with: streamlit run app.py

# Compatible with the existing backend.py:
#     discover_formula(df, target_column, effort)
#     EFFORT_PRESETS
# """

# from io import StringIO
# import json
# import numpy as np
# import pandas as pd
# import streamlit as st

# from backend import discover_formula, EFFORT_PRESETS

# # --------------------------- Page configuration ---------------------------
# st.set_page_config(
#     page_title="AI Formula Discovery",
#     page_icon="🧬",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ------------------------------- Styling (Fixed Strict High-Contrast) ----------------------------------
# st.markdown(
#     """
#     <style>
#     /* 1. Global Dark Background & White Text */
#     .stApp { 
#         background-color: #0f172a !important; 
#         color: #ffffff !important;
#     }
#     [data-testid="stHeader"] { 
#         background: rgba(15, 23, 42, 0.9) !important; 
#     }
#     .block-container { 
#         padding-top: 2rem; 
#         padding-bottom: 3rem; 
#         max-width: 1250px; 
#     }
    
#     /* 2. Headings & Markdown Text -> Always White */
#     h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { 
#         color: #ffffff !important; 
#     }

#     /* 3. Hero Banner -> Dark Gradient Background with Pure White Text */
#     .hero {
#         padding: 1.8rem 2rem; 
#         border-radius: 16px; 
#         margin-bottom: 1.5rem;
#         background: linear-gradient(135deg, #090d16 0%, #1e293b 70%, #312e81 100%) !important;
#         color: #ffffff !important; 
#         border: 1px solid rgba(255, 255, 255, 0.15);
#         box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
#     }
#     .hero h1 { 
#         color: #ffffff !important; 
#         margin: 0 0 0.5rem 0 !important; 
#         font-size: 2.2rem !important; 
#     }
#     .hero p { 
#         color: #cbd5e1 !important; 
#         font-size: 1.05rem !important; 
#         margin: 0 !important; 
#     }

#     /* 4. Metric Containers Styling -> White Card Background with Dark /* 4. Metric Containers Styling -> White Card Background with Dark High-Contrast Text */
#     div[data-testid="stMetric"] {
#         background: #ffffff !important; 
#         padding: 1.2rem !important; 
#         border: 1px solid #cbd5e1 !important;
#         border-radius: 12px !important; 
#         box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
#     }
    
#     /* Metric Label (e.g. "Rows", "Columns") -> Dark Gray */
#     div[data-testid="stMetric"] label, 
#     div[data-testid="stMetric"] [data-testid="stMetricLabel"],
#     div[data-testid="stMetric"] p {
#         color: #334155 !important;
#         font-weight: 600 !important;
#     }
    
#     /* Metric Value (e.g. "30", "4") -> Pure Black & Bold */
#     div[data-testid="stMetric"] [data-testid="stMetricValue"],
#     div[data-testid="stMetric"] div[data-testid="stMetricValue"] > div {
#         color: #020617 !important;
#         font-weight: 800 !important;
#         font-size: 1.8rem !important;
#     }

#     /* 5. Buttons */
#     div.stButton > button, div.stDownloadButton > button {
#         border-radius: 10px !important; 
#         font-weight: 650 !important; 
#         min-height: 2.8rem !important;
#         background-color: #1e293b !important;
#         color: #ffffff !important;
#         border: 1px solid #475569 !important;
#     }
#     div.stButton > button[kind="primary"] {
#         background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
#         border: 0 !important; 
#         color: #ffffff !important;
#         box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
#     }
#     div.stButton > button[kind="primary"]:hover {
#         background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
#         color: #ffffff !important;
#     }

#     /* 6. Code Blocks Styling -> Dark Background with Bright High-Contrast Text */
#     pre, code, .stCodeBlock, .stCodeBlock code {
#         background-color: #020617 !important;
#         color: #38bdf8 !important; /* Bright cyan/blue for code text */
#         border: 1px solid #334155 !important;
#         border-radius: 8px !important;
#     }
    
#     .stCodeBlock pre {
#         background-color: #020617 !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # -------------------------------- Header ----------------------------------
# st.markdown(
#     """
#     <div class="hero">
#       <h1>🧬 AI-Powered Mathematical Formula Discovery</h1>
#       <p>Discover interpretable mathematical expressions from numeric data using
#       Genetic Programming (symbolic regression).</p>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# st.caption(
#     "Workflow: upload a CSV → choose the target variable → configure the search → "
#     "discover and evaluate a candidate formula."
# )

# # ------------------------------ Upload data -------------------------------
# uploaded = st.file_uploader(
#     "Upload your dataset",
#     type=["csv"],
#     help="Use a CSV file with numeric input columns and one numeric target column.",
# )

# if uploaded is None:
#     st.info("**No dataset uploaded yet.** Upload a CSV file to preview your data and begin.")
#     with st.expander("Example CSV structure"):
#         st.code("x,temperature,output\n1,20,4.2\n2,25,7.1\n3,30,10.3", language="csv")
#     st.stop()

# try:
#     df = pd.read_csv(uploaded)
# except Exception as exc:
#     st.error(f"Could not read this CSV file. Check that it is a valid CSV. Details: {exc}")
#     st.stop()

# if df.empty:
#     st.error("The uploaded CSV has no data rows.")
#     st.stop()

# if df.columns.duplicated().any():
#     duplicate_names = df.columns[df.columns.duplicated()].astype(str).tolist()
#     st.error("Duplicate column names found: " + ", ".join(duplicate_names))
#     st.info("Rename duplicate columns in your CSV and upload it again.")
#     st.stop()

# if df.shape[1] < 2:
#     st.error("The dataset must contain at least two columns: one input feature and one target.")
#     st.stop()

# # ---------------------------- Dataset overview ----------------------------
# st.markdown("## 1. Dataset overview")
# m1, m2, m3, m4 = st.columns(4)
# m1.metric("Rows", f"{len(df):,}")
# m2.metric("Columns", f"{df.shape[1]:,}")
# m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
# numeric_cols = df.select_dtypes(include="number").columns.tolist()
# m4.metric("Numeric columns", f"{len(numeric_cols):,}")

# with st.expander("Preview uploaded data", expanded=True):
#     st.dataframe(df.head(10), use_container_width=True)

# with st.expander("Column quality report"):
#     quality = pd.DataFrame({
#         "Column": df.columns.astype(str),
#         "Data type": [str(dtype) for dtype in df.dtypes],
#         "Missing values": [int(df[col].isna().sum()) for col in df.columns],
#         "Unique values": [int(df[col].nunique(dropna=True)) for col in df.columns],
#         "Numeric": [bool(pd.api.types.is_numeric_dtype(df[col])) for col in df.columns],
#     })
#     st.dataframe(quality, use_container_width=True, hide_index=True)

# non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()
# if non_numeric_cols:
#     st.warning(
#         "Non-numeric columns detected: "
#         + ", ".join(map(str, non_numeric_cols))
#         + ". The current backend ignores non-numeric columns during formula discovery."
#     )

# # --------------------------- Model configuration --------------------------
# st.markdown("## 2. Configure formula discovery")
# left, right = st.columns([1.25, 1])

# with left:
#     target = st.selectbox(
#         "Target/output column",
#         options=list(df.columns),
#         index=len(df.columns) - 1,
#         help="Choose the numeric column whose relationship with the other numeric columns you want to discover.",
#     )
#     numeric_inputs = [c for c in numeric_cols if c != target]
#     st.write("**Numeric input columns used by the current backend**")
#     if numeric_inputs:
#         st.write(", ".join(map(str, numeric_inputs)))
#     else:
#         st.warning("No numeric input columns are available apart from the selected target.")

# with right:
#     effort = st.selectbox(
#         "Search effort",
#         options=list(EFFORT_PRESETS.keys()),
#         index=list(EFFORT_PRESETS.keys()).index("Standard")
#         if "Standard" in EFFORT_PRESETS else 0,
#         help="Higher effort usually explores more candidate expressions but takes longer.",
#     )
#     preset = EFFORT_PRESETS[effort]
#     st.caption(
#         f"Current preset: population size **{preset[0]}**, "
#         f"generations **{preset[1]}**, maximum tree depth **{preset[2]}**."
#     )

# st.warning(
#     "The current backend uses all numeric columns except the target as inputs. "
#     "Selecting individual input features requires a corresponding backend change."
# )

# # Check whether target is numeric and whether at least one numeric feature exists.
# if not pd.api.types.is_numeric_dtype(df[target]):
#     st.error("The selected target column is not numeric. Choose a numeric target column.")
#     st.stop()

# if not numeric_inputs:
#     st.error("At least one numeric input column is required, in addition to the target.")
#     st.stop()

# # Show a lightweight preview plot of target values (not model predictions).
# with st.expander("Explore target column"):
#     target_values = pd.to_numeric(df[target], errors="coerce").dropna()
#     if len(target_values):
#         st.line_chart(target_values.reset_index(drop=True).rename("Target value"))
#     else:
#         st.info("No valid numeric target values are available to plot.")

# # ---------------------------- Run discovery -------------------------------
# st.markdown("## 3. Discover a formula")
# st.write(
#     "The search may take some time depending on the dataset size and selected effort. "
#     "The discovered expression is a candidate explanation, not a guarantee that the original formula has been recovered."
# )

# run = st.button("🔍 Discover Formula", type="primary", use_container_width=True)

# if run:
#     # Basic UI-side validation; backend performs its own cleaning and checks too.
#     numeric_frame = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
#     usable = numeric_frame.dropna(subset=[target])
#     if len(usable) < 5:
#         st.error("At least five rows with valid numeric target values are required.")
#         st.stop()

#     try:
#         with st.spinner("🧬 Genetic Programming is evolving candidate expressions..."):
#             result = discover_formula(df, target, effort)
#         st.session_state["formula_result"] = result
#         st.session_state["formula_target_original"] = str(target)
#         st.session_state["formula_effort"] = effort
#     except ValueError as exc:
#         st.error(str(exc))
#         st.stop()
#     except Exception as exc:
#         st.error("Formula discovery failed. Check the dataset and try again.")
#         with st.expander("Technical details (for debugging)"):
#             st.code(f"{type(exc).__name__}: {exc}")
#         st.stop()

# # ------------------------------ Results -----------------------------------
# result = st.session_state.get("formula_result")
# if result:
#     st.markdown("---")
#     st.markdown("## 4. Discovery results")
#     st.success("Formula discovery completed.")

#     st.markdown("### Best discovered expression")
#     st.code(f"{result.get('target', target)} = {result.get('formula', 'Formula unavailable')}", language="text")

#     a, b, c, d = st.columns(4)
#     r2_value = result.get("r2")
#     mse_value = result.get("mse")
#     mae_value = result.get("mae")
#     complexity_value = result.get("complexity")

#     a.metric("R² score", f"{r2_value:.5f}" if r2_value is not None and np.isfinite(r2_value) else "N/A")
#     b.metric("MAE", f"{mae_value:.5g}" if mae_value is not None and np.isfinite(mae_value) else "N/A")
#     c.metric("MSE", f"{mse_value:.5g}" if mse_value is not None and np.isfinite(mse_value) else "N/A")
#     d.metric("Formula complexity", f"{complexity_value} nodes" if complexity_value is not None else "N/A")

#     st.markdown("### Dataset processing summary")
#     s1, s2, s3 = st.columns(3)
#     s1.metric("Valid rows used", f"{result.get('n_rows', 'N/A')}")
#     s2.metric("Rows dropped", f"{result.get('n_dropped', 0)}")
#     s3.metric("Unique expressions explored", f"{result.get('n_explored', 'N/A')}")

#     ignored = result.get("ignored_columns", [])
#     if ignored:
#         st.info("Ignored non-numeric columns: " + ", ".join(map(str, ignored)))
#     if result.get("n_dropped", 0):
#         st.warning(
#             f"{result['n_dropped']} row(s) were removed because they contained missing or infinite "
#             "values in the columns used by the backend."
#         )

#     if r2_value is not None and np.isfinite(r2_value):
#         if r2_value >= 0.95:
#             st.success("The formula has a strong fit on the rows evaluated by the current backend.")
#         elif r2_value >= 0.70:
#             st.info("The formula has a moderate fit. Try a higher search effort or inspect the data for noise.")
#         else:
#             st.warning(
#                 "The fit is limited. Consider checking the target/features, data quality, noise, "
#                 "and search effort."
#             )

#     st.caption(
#         "Important: the existing backend calculates metrics on the cleaned dataset used by its workflow. "
#         "These are not independent test-set metrics; a separate train/test workflow would require backend changes."
#     )

#     # Download a report using only values returned by the existing backend.
#     report = {
#         "target": result.get("target"),
#         "formula": result.get("formula"),
#         "raw_formula": result.get("raw_formula"),
#         "input_variables": result.get("variables", []),
#         "search_effort": st.session_state.get("formula_effort", effort),
#         "mse": result.get("mse"),
#         "mae": result.get("mae"),
#         "r2": result.get("r2"),
#         "complexity_nodes": result.get("complexity"),
#         "rows_used": result.get("n_rows"),
#         "rows_dropped": result.get("n_dropped"),
#         "unique_expressions_explored": result.get("n_explored"),
#         "note": "Metrics are calculated by the current backend and are not independent test-set metrics.",
#     }
#     st.markdown("### 5. Export results")
#     report_csv = pd.DataFrame([{
#         **{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in report.items()}
#     }]).to_csv(index=False).encode("utf-8")
#     report_json = json.dumps(report, indent=2, default=str).encode("utf-8")

#     dl1, dl2 = st.columns(2)
#     with dl1:
#         st.download_button(
#             "⬇️ Download formula report (CSV)",
#             data=report_csv,
#             file_name="formula_discovery_report.csv",
#             mime="text/csv",
#             use_container_width=True,
#         )
#     with dl2:
#         st.download_button(
#             "⬇️ Download formula report (JSON)",
#             data=report_json,
#             file_name="formula_discovery_report.json",
#             mime="application/json",
#             use_container_width=True,
#         )

#     with st.expander("Technical result details"):
#         st.json(report)

# st.markdown("---")
# st.markdown(
#     "<p class='subtle' style='text-align:center'>AI Formula Discovery · Genetic Programming / Symbolic Regression</p>",
#     unsafe_allow_html=True,
# )


"""
app.py  -  Streamlit UI
------------------------
Run with:

    streamlit run app.py

Flow:
    1. Upload a CSV
    2. Review the automatic data-quality check
    3. Pick the target column, the input features and the test-set size
    4. Click "Discover Formula" and watch the Genetic Programming evolve live
    5. Review the formula, metrics, Top-5 formulas, charts and download results

All real work (cleaning, GP search, simplification, metrics) lives in backend.py
and the GP modules. This file only displays what the backend returns.
"""

import io
import html
import datetime

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from backend import (discover_formula, validate_dataframe, count_unusable_rows,
                     EFFORT_PRESETS)
from node import tree_to_string, count_nodes

st.set_page_config(page_title="AI Formula Discovery", page_icon="🧬", layout="wide")

NAVY, PURPLE, AMBER, GRID = "#1B1F5E", "#6C4DF6", "#F59E0B", "#E3E5F3"

# ----------------------------------------------------------------------------
# Theme (navy / purple) - injected CSS
# ----------------------------------------------------------------------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { background:#F4F5FB; color:#161B45;
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; }
.block-container { max-width:1100px; padding-top:4.6rem; padding-bottom:4rem; }
h1,h2,h3,h4 { font-family:'Inter',-apple-system,'Segoe UI',sans-serif; color:#14183E; letter-spacing:-0.01em; }
#MainMenu, footer { visibility:hidden; }

.hero { background:linear-gradient(120deg,#0F1445 0%,#2A2A8C 55%,#6C4DF6 100%); border-radius:20px;
  padding:34px 38px; color:#fff; margin-bottom:22px; box-shadow:0 10px 30px rgba(31,34,110,.25); }
.hero h1 { color:#fff; font-size:2.1rem; font-weight:800; margin:0 0 6px 0; padding:0; }
.hero p { color:#D8D9F7; font-size:1.02rem; margin:0; max-width:760px; line-height:1.55; }
.hero .pill { display:inline-block; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25);
  color:#fff; border-radius:999px; padding:3px 12px; font-size:.78rem; font-weight:600; margin:14px 8px 0 0; }

.step { display:flex; align-items:center; gap:12px; font-weight:700; font-size:1.12rem; color:#14183E; margin:2px 0 10px 0; }
.step .num { background:linear-gradient(135deg,#2A2A8C,#6C4DF6); color:#fff; width:30px; height:30px; border-radius:50%;
  display:inline-flex; align-items:center; justify-content:center; font-size:.9rem; font-weight:700; }
.sub { color:#5B6085; font-size:.92rem; margin:-4px 0 12px 42px; }

div[class*="st-key-card_"], div[data-testid="stVerticalBlockBorderWrapper"] { background:#fff !important;
  border-radius:16px; border:1px solid #E3E5F3 !important; box-shadow:0 2px 12px rgba(27,31,94,.06); padding:6px 8px; }

.stat { background:#F7F7FF; border:1px solid #E3E5F3; border-radius:14px; padding:14px 16px; min-height:104px; }
.stat .l { color:#5B6085; font-size:.76rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
.stat .v { color:#14183E; font-size:1.6rem; font-weight:800; line-height:1.25; }
.stat .v.warn { color:#C2680A; } .stat .v.ok { color:#3B2FC9; }
.stat .s { color:#7A7FA3; font-size:.78rem; }

.formula { background:linear-gradient(135deg,#14183E 0%,#2A2A8C 60%,#5B3FE0 100%); border-radius:18px; padding:26px 30px;
  color:#fff; box-shadow:0 10px 28px rgba(31,34,110,.28); margin:6px 0 18px 0; }
.formula .l { color:#BFC2F5; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.09em; }
.formula .f { font-family:'JetBrains Mono','Fira Code',Consolas,Menlo,monospace; font-size:1.35rem; font-weight:600;
  margin-top:8px; line-height:1.5; word-break:break-word; }
.formula .m { color:#CFD1FA; font-size:.84rem; margin-top:12px; }

.metric { background:#fff; border:1px solid #E3E5F3; border-left:5px solid #6C4DF6; border-radius:14px; padding:14px 18px;
  box-shadow:0 2px 8px rgba(27,31,94,.05); }
.metric.test { border-left-color:#F59E0B; }
.metric .l { color:#5B6085; font-size:.76rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
.metric .v { color:#14183E; font-size:1.55rem; font-weight:800; }
.group { font-weight:700; color:#14183E; margin:14px 0 8px 0; font-size:1rem; }

.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background:linear-gradient(120deg,#2A2A8C,#6C4DF6); border:none; color:#fff; font-weight:700; border-radius:12px;
  padding:.65rem 1.2rem; box-shadow:0 6px 16px rgba(108,77,246,.35); }
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { filter:brightness(1.08); color:#fff; }
.stDownloadButton button { border-radius:12px; border:1.5px solid #6C4DF6; color:#2A2A8C; font-weight:600; background:#fff; }
.stDownloadButton button:hover { background:#F1EEFF; color:#2A2A8C; border-color:#6C4DF6; }
button[data-baseweb="tab"] { font-weight:600; }
.stProgress > div > div > div > div { background-color:#6C4DF6; }
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Small display helpers
# ----------------------------------------------------------------------------
def fmt(x, digits=4):
    """Format a number; show 'n/a' for NaN / infinity."""
    try:
        return f"{x:.{digits}f}" if np.isfinite(x) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def step_header(num, title, subtitle=""):
    st.markdown(f'<div class="step"><span class="num">{num}</span>{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def stat_card(label, value, sub="\u00a0", cls=""):
    return (f'<div class="stat"><div class="l">{html.escape(label)}</div>'
            f'<div class="v {cls}">{html.escape(str(value))}</div><div class="s">{html.escape(sub)}</div></div>')


def metric_card(label, value, cls=""):
    return (f'<div class="metric {cls}"><div class="l">{html.escape(label)}</div>'
            f'<div class="v">{html.escape(value)}</div></div>')


@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


# ----------------------------------------------------------------------------
# Charts (matplotlib Figure objects, styled to match the theme)
# ----------------------------------------------------------------------------
def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color="#3A3F6B")
    ax.set_ylabel(ylabel, color="#3A3F6B")
    ax.set_title(title, color=NAVY, fontweight="bold", fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C9CCE6")
    ax.tick_params(colors="#3A3F6B")


def _finite(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def fig_actual_vs_predicted(res):
    fig = Figure(figsize=(9, 4.8))
    ax = fig.subplots()
    ya, pa = _finite(res["y_train"], res["pred_train"])
    ax.scatter(ya, pa, s=26, alpha=0.75, color=PURPLE, label="Train", edgecolor="white", linewidth=0.4)
    lows, highs = [ya.min()], [ya.max()]
    if res["y_test"] is not None:
        yt, pt = _finite(res["y_test"], res["pred_test"])
        ax.scatter(yt, pt, s=34, alpha=0.9, color=AMBER, label="Test", edgecolor="white", linewidth=0.4)
        lows.append(yt.min()); highs.append(yt.max())
    lo, hi = min(lows), max(highs)
    ax.plot([lo, hi], [lo, hi], "--", color=NAVY, linewidth=1.2, label="Perfect prediction")
    _style(ax, "Actual", "Predicted", "Actual vs predicted")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_errors(res):
    fig = Figure(figsize=(10, 4.2))
    ax1, ax2 = fig.subplots(1, 2)
    e_tr = np.asarray(res["y_train"], float) - np.asarray(res["pred_train"], float)
    e_tr = e_tr[np.isfinite(e_tr)]
    has_test = res["y_test"] is not None
    ax1.hist(e_tr, bins=20, color=PURPLE, alpha=0.8, label="Train", density=has_test, edgecolor="white")
    if has_test:
        e_te = np.asarray(res["y_test"], float) - np.asarray(res["pred_test"], float)
        e_te = e_te[np.isfinite(e_te)]
        ax1.hist(e_te, bins=12, color=AMBER, alpha=0.7, label="Test", density=True, edgecolor="white")
    ax1.axvline(0, color=NAVY, linewidth=1.2, linestyle="--")
    _style(ax1, "Error (actual - predicted)", "Density" if has_test else "Count", "Error distribution")
    ax1.legend(frameon=False)

    p, e = _finite(res["pred_train"], np.asarray(res["y_train"], float) - np.asarray(res["pred_train"], float))
    ax2.scatter(p, e, s=24, alpha=0.75, color=PURPLE, label="Train", edgecolor="white", linewidth=0.4)
    if has_test:
        p2, e2 = _finite(res["pred_test"], np.asarray(res["y_test"], float) - np.asarray(res["pred_test"], float))
        ax2.scatter(p2, e2, s=32, alpha=0.9, color=AMBER, label="Test", edgecolor="white", linewidth=0.4)
    ax2.axhline(0, color=NAVY, linewidth=1.2, linestyle="--")
    _style(ax2, "Predicted", "Error", "Errors vs predicted value")
    ax2.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_fitness(res):
    hist = np.asarray(res["history"], float)
    fig = Figure(figsize=(9, 4.2))
    ax = fig.subplots()
    gens = np.arange(1, len(hist) + 1)
    ax.plot(gens, hist, color=PURPLE, linewidth=2.2)
    ax.fill_between(gens, hist, hist.min(), color=PURPLE, alpha=0.08)
    if hist.min() > 0 and hist.max() / hist.min() > 50:
        ax.set_yscale("log")
    _style(ax, "Generation", "Best fitness (lower is better)", "Fitness across generations")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Formula report (built only from real results)
# ----------------------------------------------------------------------------
def build_report(res, source_name):
    s = res["settings"]
    lines = [
        "# Formula Discovery Report", "",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Data file: {source_name}", "",
        "## Discovered formula", "",
        f"    {res['target']} = {res['formula']}", "",
        f"Raw Genetic Programming expression: `{res['raw_formula']}`", "",
        f"Complexity: {res['complexity']} nodes", "",
        "## Performance", "",
        "| Split | Rows | R2 | MAE | MSE |", "|---|---|---|---|---|",
        f"| Training | {s['n_train']} | {fmt(res['train_metrics']['r2'], 6)} | "
        f"{fmt(res['train_metrics']['mae'], 6)} | {fmt(res['train_metrics']['mse'], 6)} |",
    ]
    if res["test_metrics"] is not None:
        t = res["test_metrics"]
        lines.append(f"| Testing | {s['n_test']} | {fmt(t['r2'], 6)} | {fmt(t['mae'], 6)} | {fmt(t['mse'], 6)} |")
    else:
        lines.append("| Testing | - | no test set was held out | | |")
    lines += ["", "## Top formulas", ""]
    has_test = res["test_metrics"] is not None
    lines.append("| Rank | Formula | Train MSE | " + ("Test MSE | " if has_test else "") + "Complexity |")
    lines.append("|---|---|---|" + ("---|" if has_test else "") + "---|")
    for i, f in enumerate(res["top_formulas"], 1):
        row = f"| {i} | `{f['formula']}` | {fmt(f['train_mse'], 6)} | "
        if has_test:
            row += f"{fmt(f['test_mse'], 6)} | "
        lines.append(row + f"{f['complexity']} |")
    lines += [
        "", "## Data", "",
        f"- Target column: {res['target']}",
        f"- Input features: {', '.join(res['variables'])}",
        f"- Rows used: {res['n_rows']} (removed for missing/infinite values: {res['n_dropped']})",
        f"- Training rows: {s['n_train']}, test rows: {s['n_test']}",
        f"- Non-numeric columns ignored: {', '.join(res['ignored_columns']) or 'none'}", "",
        "## Genetic Programming settings", "",
        f"- Search effort: {s['effort']} (population {s['population']}, {s['generations']} generations, "
        f"max tree depth {s['max_depth']})",
        f"- Rows used by the search itself: {s['n_search_rows']}",
        f"- Random seed: {s['seed']}",
        f"- Unique formulas explored: {res['n_explored']}",
        f"- Best fitness in final generation: {fmt(res['history'][-1], 6)} (MSE + 0.01 x complexity)",
        f"- Run time: {res['elapsed_seconds']:.1f} s", "",
        "Note: Genetic Programming is a stochastic search. Equivalent formulas can look different, "
        "and a different seed or settings may give a different (but similarly accurate) formula.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Results section
# ----------------------------------------------------------------------------
def render_results(res, source_name):
    s = res["settings"]
    st.markdown(f"""<div class="formula"><div class="l">Discovered formula</div>
<div class="f">{html.escape(res['target'])} = {html.escape(res['formula'])}</div>
<div class="m">Complexity {res['complexity']} nodes &nbsp;•&nbsp; {res['n_explored']:,} unique formulas explored
&nbsp;•&nbsp; {res['elapsed_seconds']:.1f}s</div></div>""".replace("\n", " "), unsafe_allow_html=True)

    with st.expander("Show raw Genetic Programming expression"):
        st.code(res["raw_formula"], language="text")

    # ---- metrics ----
    st.markdown('<div class="group">Training performance <span style="color:#7A7FA3;font-weight:500">'
                f'({s["n_train"]} rows)</span></div>', unsafe_allow_html=True)
    tm = res["train_metrics"]
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("R² score", fmt(tm["r2"], 5)), unsafe_allow_html=True)
    c2.markdown(metric_card("MAE", fmt(tm["mae"])), unsafe_allow_html=True)
    c3.markdown(metric_card("MSE", fmt(tm["mse"])), unsafe_allow_html=True)

    if res["test_metrics"] is not None:
        te = res["test_metrics"]
        st.markdown('<div class="group">Testing performance <span style="color:#7A7FA3;font-weight:500">'
                    f'({s["n_test"]} unseen rows)</span></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("R² score", fmt(te["r2"], 5), "test"), unsafe_allow_html=True)
        c2.markdown(metric_card("MAE", fmt(te["mae"]), "test"), unsafe_allow_html=True)
        c3.markdown(metric_card("MSE", fmt(te["mse"]), "test"), unsafe_allow_html=True)
        if np.isfinite(te["r2"]) and np.isfinite(tm["r2"]) and tm["r2"] - te["r2"] > 0.1:
            st.warning("Performance drops noticeably on the unseen test rows, so this formula may be "
                       "overfitting. Try more data or a simpler search.")
    else:
        st.info("No test set was held out (test split = 0%), so only training performance is shown.")

    if np.isfinite(tm["r2"]) and tm["r2"] < 0.95:
        st.info("The fit is not very tight. Try a higher search effort under 'Advanced settings', "
                "or the data may not follow a simple formula.")

    # ---- top 5 ----
    st.markdown('<div class="group" style="margin-top:22px">Top 5 discovered formulas</div>', unsafe_allow_html=True)
    rows = []
    for i, f in enumerate(res["top_formulas"], 1):
        row = {"Rank": i, "Formula": f["formula"], "Train MSE": f["train_mse"]}
        if res["test_metrics"] is not None:
            row["Test MSE"] = f["test_mse"]
        row["Complexity"] = f["complexity"]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "Formula": st.column_config.TextColumn("Formula", width="large")})
    st.caption("Ranked by GP fitness (MSE + 0.01 × complexity) on the training sample. "
               "Formulas that simplify to the same expression are shown once.")
    with st.expander("Show full formulas"):
        for i, f in enumerate(res["top_formulas"], 1):
            st.code(f"{i}. {res['target']} = {f['formula']}", language="text")

    # ---- charts ----
    st.markdown('<div class="group" style="margin-top:22px">Visualizations</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Actual vs predicted", "Prediction errors", "Fitness across generations"])
    with t1:
        st.pyplot(fig_actual_vs_predicted(res))
    with t2:
        st.pyplot(fig_errors(res))
    with t3:
        st.pyplot(fig_fitness(res))

    # ---- downloads ----
    st.markdown('<div class="group" style="margin-top:22px">Download results</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇  Predictions (CSV)", data=res["predictions_df"].to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{res['target']}.csv", mime="text/csv")
    d2.download_button("⬇  Formula report (Markdown)", data=build_report(res, source_name).encode("utf-8"),
                       file_name=f"formula_report_{res['target']}.md", mime="text/markdown")


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.markdown("""<div class="hero"><h1>🧬 AI Formula Discovery</h1>
<p>Upload a CSV and the AI discovers the mathematical formula hidden in your data using Genetic Programming,
an evolutionary search built from scratch. No gradient descent, no ready-made symbolic regression library.</p>
<span class="pill">Expression trees</span><span class="pill">Crossover &amp; mutation</span>
<span class="pill">Live evolution</span></div>""".replace("\n", " "), unsafe_allow_html=True)

with st.container(border=True, key="card_upload"):
    step_header(1, "Upload your data", "A CSV file with numeric input columns and one numeric output column.")
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

if uploaded is None:
    st.info("👆 Upload a CSV to get started. Sample files: sample_data/ohms_law.csv, sample_data/kinetic_energy.csv")
    st.stop()

try:
    df = load_csv(uploaded.getvalue())
except Exception:
    st.error("Could not read this file. Please upload a valid CSV.")
    st.stop()

if df.shape[1] < 2:
    st.error("The CSV needs at least 2 columns (inputs + one output column).")
    st.stop()

# ---------------- Step 2: dataset validation ----------------
report = validate_dataframe(df)
with st.container(border=True, key="card_check"):
    step_header(2, "Data check", "Automatic validation of your file before anything runs.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(stat_card("Rows", f"{report['n_rows']:,}", "in your file"), unsafe_allow_html=True)
    c2.markdown(stat_card("Columns", report["n_cols"], f"{len(report['numeric_columns'])} numeric"), unsafe_allow_html=True)
    c3.markdown(stat_card("Missing values", report["total_missing"],
                          f"in {len(report['missing_by_column'])} column(s)",
                          "warn" if report["total_missing"] else "ok"), unsafe_allow_html=True)
    c4.markdown(stat_card("Infinite values", report["total_inf"],
                          f"in {len(report['inf_by_column'])} column(s)",
                          "warn" if report["total_inf"] else "ok"), unsafe_allow_html=True)
    c5.markdown(stat_card("Non-numeric cols", len(report["non_numeric_columns"]), "will be ignored",
                          "warn" if report["non_numeric_columns"] else "ok"), unsafe_allow_html=True)
    st.write("")

    if report["name_clashes"]:
        st.error("These columns have names that become identical after cleaning: "
                 + ", ".join(report["name_clashes"]) + ". Please rename them in your CSV.")
        st.stop()
    if len(report["numeric_columns"]) < 2:
        st.error("Need at least 2 numeric columns (one input and one output). "
                 "Non-numeric columns found: " + (", ".join(report["non_numeric_columns"]) or "none"))
        st.stop()
    if report["non_numeric_columns"]:
        st.info("Text / non-numeric columns are ignored: " + ", ".join(report["non_numeric_columns"]))
    if report["missing_by_column"]:
        st.warning("Missing values found (" + ", ".join(f"{k}: {v}" for k, v in report["missing_by_column"].items())
                   + "). Rows with a missing value in the selected columns are removed.")
    if report["inf_by_column"]:
        st.warning("Infinite values found (" + ", ".join(f"{k}: {v}" for k, v in report["inf_by_column"].items())
                   + "). Rows with an infinite value in the selected columns are removed.")
    if not (report["missing_by_column"] or report["inf_by_column"] or report["non_numeric_columns"]):
        st.success("No missing, infinite or non-numeric values found.")
    with st.expander("Preview data (first 10 rows)"):
        st.dataframe(df.head(10))

# ---------------- Step 3: configuration ----------------
numeric_cols = report["numeric_columns"]
with st.container(border=True, key="card_columns"):
    step_header(3, "Choose columns", "Pick what to predict, which columns the formula may use, and how much data to hold out.")
    left, right = st.columns([3, 2])
    with left:
        target = st.selectbox("Output column (the one to find a formula for)", numeric_cols,
                              index=len(numeric_cols) - 1)
    with right:
        test_pct = st.slider("Hold-out test set (% of rows)", 0, 40, 20, step=5,
                             help="These rows are never shown to the Genetic Programming search. "
                                  "They are used to check how well the formula works on unseen data.")
    feature_options = [c for c in numeric_cols if c != target]
    features = st.multiselect("Input features (columns the formula may use)", feature_options,
                              default=feature_options, key=f"features_{target}")
    with st.expander("Advanced settings"):
        effort = st.select_slider("Search effort (higher = slower but better for complex formulas)",
                                  options=list(EFFORT_PRESETS.keys()), value="Standard")
        pop, gens, depth = EFFORT_PRESETS[effort]
        st.caption(f"{effort}: population {pop}, {gens} generations, max tree depth {depth}.")

    # Pre-flight checks that depend on the selection
    problems = []
    if report["constant_columns"] and target in report["constant_columns"]:
        problems.append("The output column has the same value in every row, so there is no formula to find.")
    if not features:
        problems.append("Select at least one input feature.")
    usable = len(df)
    if features:
        removed = count_unusable_rows(df, features + [target])
        usable = len(df) - removed
        n_test = max(2, int(round(usable * test_pct / 100))) if test_pct > 0 else 0
        msg = f"{usable:,} of {len(df):,} rows are usable"
        if removed:
            msg += f" ({removed} removed for missing/infinite values)"
        if test_pct > 0 and usable - n_test >= 5:
            msg += f" → about {usable - n_test:,} for training and {n_test:,} for testing."
        st.caption(msg)
        if usable < 5:
            problems.append("Fewer than 5 usable rows remain after cleaning.")
        elif test_pct > 0 and usable - n_test < 5:
            problems.append("Too few rows for this test split. Lower the test percentage.")
    const_in = [f for f in features if f in report["constant_columns"]]
    if const_in:
        st.info("Constant columns carry no information: " + ", ".join(const_in)
                + ". You may want to remove them from the input features.")
    for p in problems:
        st.error(p)

    run_clicked = st.button("🔍 Discover Formula", type="primary", disabled=bool(problems))

# ---------------- Step 4: run with live progress ----------------
run_key = (uploaded.name, len(uploaded.getvalue()), target, tuple(features), effort, test_pct)

if run_clicked:
    with st.container(border=True, key="card_run"):
        step_header(4, "Evolving formulas", "Live progress from the Genetic Programming engine.")
        bar = st.progress(0.0, text="Initialising population...")
        k1, k2, k3 = st.columns(3)
        ph_gen, ph_fit, ph_size = k1.empty(), k2.empty(), k3.empty()
        ph_expr = st.empty()
        ph_chart = st.empty()
        live = []

        def on_generation(g, total, best_fitness, best_tree):
            live.append(float(np.log10(max(best_fitness, 1e-12))))
            bar.progress(g / total, text=f"Generation {g} of {total}")
            ph_gen.markdown(stat_card("Generation", f"{g} / {total}"), unsafe_allow_html=True)
            ph_fit.markdown(stat_card("Best fitness", f"{best_fitness:.6g}", "lower is better"), unsafe_allow_html=True)
            ph_size.markdown(stat_card("Best formula size", f"{count_nodes(best_tree)} nodes"), unsafe_allow_html=True)
            expr = tree_to_string(best_tree)
            ph_expr.code("best so far: " + (expr if len(expr) <= 110 else expr[:107] + "..."), language="text")
            if g % 2 == 0 or g == total:
                ph_chart.line_chart(pd.DataFrame({"log10(best fitness), lower is better": live}), height=190, color="#6C4DF6")

        try:
            result = discover_formula(df, target, effort, features=features,
                                      test_size=test_pct / 100.0, progress_callback=on_generation)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error("Something went wrong while processing this data. "
                     "Please check that the selected columns contain only numeric data.")
            st.caption(f"Technical detail: {type(e).__name__}")
            st.stop()
        bar.progress(1.0, text=f"Done in {result['elapsed_seconds']:.1f}s")

    st.session_state["result"] = result
    st.session_state["result_key"] = run_key
    st.session_state["result_source"] = uploaded.name

# ---------------- Step 5: results (kept in session so downloads don't wipe them) ----------------
if "result" in st.session_state:
    if st.session_state.get("result_key") == run_key:
        with st.container(border=True, key="card_results"):
            step_header(5, "Results")
            render_results(st.session_state["result"], st.session_state.get("result_source", "uploaded file"))
    else:
        st.info("Your selections changed. Click **Discover Formula** to run again with the new settings.")