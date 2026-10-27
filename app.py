"""
app.py — Streamlit UI for AI-Based Mathematical Formula Discovery
Run with: streamlit run app.py

Compatible with the existing backend.py:
    discover_formula(df, target_column, effort)
    EFFORT_PRESETS
"""

from io import StringIO
import json
import numpy as np
import pandas as pd
import streamlit as st

from backend import discover_formula, EFFORT_PRESETS

# --------------------------- Page configuration ---------------------------
st.set_page_config(
    page_title="AI Formula Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------- Styling (Fixed Strict High-Contrast) ----------------------------------
st.markdown(
    """
    <style>
    /* 1. Global Dark Background & White Text */
    .stApp { 
        background-color: #0f172a !important; 
        color: #ffffff !important;
    }
    [data-testid="stHeader"] { 
        background: rgba(15, 23, 42, 0.9) !important; 
    }
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 3rem; 
        max-width: 1250px; 
    }
    
    /* 2. Headings & Markdown Text -> Always White */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { 
        color: #ffffff !important; 
    }

    /* 3. Hero Banner -> Dark Gradient Background with Pure White Text */
    .hero {
        padding: 1.8rem 2rem; 
        border-radius: 16px; 
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #090d16 0%, #1e293b 70%, #312e81 100%) !important;
        color: #ffffff !important; 
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }
    .hero h1 { 
        color: #ffffff !important; 
        margin: 0 0 0.5rem 0 !important; 
        font-size: 2.2rem !important; 
    }
    .hero p { 
        color: #cbd5e1 !important; 
        font-size: 1.05rem !important; 
        margin: 0 !important; 
    }

    /* 4. Metric Containers Styling -> White Card Background with Dark /* 4. Metric Containers Styling -> White Card Background with Dark High-Contrast Text */
    div[data-testid="stMetric"] {
        background: #ffffff !important; 
        padding: 1.2rem !important; 
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Metric Label (e.g. "Rows", "Columns") -> Dark Gray */
    div[data-testid="stMetric"] label, 
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] p {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    
    /* Metric Value (e.g. "30", "4") -> Pure Black & Bold */
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] > div {
        color: #020617 !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }

    /* 5. Buttons */
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px !important; 
        font-weight: 650 !important; 
        min-height: 2.8rem !important;
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        border: 0 !important; 
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
    }

    /* 6. Code Blocks Styling -> Dark Background with Bright High-Contrast Text */
    pre, code, .stCodeBlock, .stCodeBlock code {
        background-color: #020617 !important;
        color: #38bdf8 !important; /* Bright cyan/blue for code text */
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    .stCodeBlock pre {
        background-color: #020617 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------- Header ----------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🧬 AI-Powered Mathematical Formula Discovery</h1>
      <p>Discover interpretable mathematical expressions from numeric data using
      Genetic Programming (symbolic regression).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Workflow: upload a CSV → choose the target variable → configure the search → "
    "discover and evaluate a candidate formula."
)

# ------------------------------ Upload data -------------------------------
uploaded = st.file_uploader(
    "Upload your dataset",
    type=["csv"],
    help="Use a CSV file with numeric input columns and one numeric target column.",
)

if uploaded is None:
    st.info("**No dataset uploaded yet.** Upload a CSV file to preview your data and begin.")
    with st.expander("Example CSV structure"):
        st.code("x,temperature,output\n1,20,4.2\n2,25,7.1\n3,30,10.3", language="csv")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read this CSV file. Check that it is a valid CSV. Details: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV has no data rows.")
    st.stop()

if df.columns.duplicated().any():
    duplicate_names = df.columns[df.columns.duplicated()].astype(str).tolist()
    st.error("Duplicate column names found: " + ", ".join(duplicate_names))
    st.info("Rename duplicate columns in your CSV and upload it again.")
    st.stop()

if df.shape[1] < 2:
    st.error("The dataset must contain at least two columns: one input feature and one target.")
    st.stop()

# ---------------------------- Dataset overview ----------------------------
st.markdown("## 1. Dataset overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{df.shape[1]:,}")
m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
m4.metric("Numeric columns", f"{len(numeric_cols):,}")

with st.expander("Preview uploaded data", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("Column quality report"):
    quality = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data type": [str(dtype) for dtype in df.dtypes],
        "Missing values": [int(df[col].isna().sum()) for col in df.columns],
        "Unique values": [int(df[col].nunique(dropna=True)) for col in df.columns],
        "Numeric": [bool(pd.api.types.is_numeric_dtype(df[col])) for col in df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()
if non_numeric_cols:
    st.warning(
        "Non-numeric columns detected: "
        + ", ".join(map(str, non_numeric_cols))
        + ". The current backend ignores non-numeric columns during formula discovery."
    )

# --------------------------- Model configuration --------------------------
st.markdown("## 2. Configure formula discovery")
left, right = st.columns([1.25, 1])

with left:
    target = st.selectbox(
        "Target/output column",
        options=list(df.columns),
        index=len(df.columns) - 1,
        help="Choose the numeric column whose relationship with the other numeric columns you want to discover.",
    )
    numeric_inputs = [c for c in numeric_cols if c != target]
    st.write("**Numeric input columns used by the current backend**")
    if numeric_inputs:
        st.write(", ".join(map(str, numeric_inputs)))
    else:
        st.warning("No numeric input columns are available apart from the selected target.")

with right:
    effort = st.selectbox(
        "Search effort",
        options=list(EFFORT_PRESETS.keys()),
        index=list(EFFORT_PRESETS.keys()).index("Standard")
        if "Standard" in EFFORT_PRESETS else 0,
        help="Higher effort usually explores more candidate expressions but takes longer.",
    )
    preset = EFFORT_PRESETS[effort]
    st.caption(
        f"Current preset: population size **{preset[0]}**, "
        f"generations **{preset[1]}**, maximum tree depth **{preset[2]}**."
    )

st.warning(
    "The current backend uses all numeric columns except the target as inputs. "
    "Selecting individual input features requires a corresponding backend change."
)

# Check whether target is numeric and whether at least one numeric feature exists.
if not pd.api.types.is_numeric_dtype(df[target]):
    st.error("The selected target column is not numeric. Choose a numeric target column.")
    st.stop()

if not numeric_inputs:
    st.error("At least one numeric input column is required, in addition to the target.")
    st.stop()

# Show a lightweight preview plot of target values (not model predictions).
with st.expander("Explore target column"):
    target_values = pd.to_numeric(df[target], errors="coerce").dropna()
    if len(target_values):
        st.line_chart(target_values.reset_index(drop=True).rename("Target value"))
    else:
        st.info("No valid numeric target values are available to plot.")

# ---------------------------- Run discovery -------------------------------
st.markdown("## 3. Discover a formula")
st.write(
    "The search may take some time depending on the dataset size and selected effort. "
    "The discovered expression is a candidate explanation, not a guarantee that the original formula has been recovered."
)

run = st.button("🔍 Discover Formula", type="primary", use_container_width=True)

if run:
    # Basic UI-side validation; backend performs its own cleaning and checks too.
    numeric_frame = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    usable = numeric_frame.dropna(subset=[target])
    if len(usable) < 5:
        st.error("At least five rows with valid numeric target values are required.")
        st.stop()

    try:
        with st.spinner("🧬 Genetic Programming is evolving candidate expressions..."):
            result = discover_formula(df, target, effort)
        st.session_state["formula_result"] = result
        st.session_state["formula_target_original"] = str(target)
        st.session_state["formula_effort"] = effort
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error("Formula discovery failed. Check the dataset and try again.")
        with st.expander("Technical details (for debugging)"):
            st.code(f"{type(exc).__name__}: {exc}")
        st.stop()

# ------------------------------ Results -----------------------------------
result = st.session_state.get("formula_result")
if result:
    st.markdown("---")
    st.markdown("## 4. Discovery results")
    st.success("Formula discovery completed.")

    st.markdown("### Best discovered expression")
    st.code(f"{result.get('target', target)} = {result.get('formula', 'Formula unavailable')}", language="text")

    a, b, c, d = st.columns(4)
    r2_value = result.get("r2")
    mse_value = result.get("mse")
    mae_value = result.get("mae")
    complexity_value = result.get("complexity")

    a.metric("R² score", f"{r2_value:.5f}" if r2_value is not None and np.isfinite(r2_value) else "N/A")
    b.metric("MAE", f"{mae_value:.5g}" if mae_value is not None and np.isfinite(mae_value) else "N/A")
    c.metric("MSE", f"{mse_value:.5g}" if mse_value is not None and np.isfinite(mse_value) else "N/A")
    d.metric("Formula complexity", f"{complexity_value} nodes" if complexity_value is not None else "N/A")

    st.markdown("### Dataset processing summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Valid rows used", f"{result.get('n_rows', 'N/A')}")
    s2.metric("Rows dropped", f"{result.get('n_dropped', 0)}")
    s3.metric("Unique expressions explored", f"{result.get('n_explored', 'N/A')}")

    ignored = result.get("ignored_columns", [])
    if ignored:
        st.info("Ignored non-numeric columns: " + ", ".join(map(str, ignored)))
    if result.get("n_dropped", 0):
        st.warning(
            f"{result['n_dropped']} row(s) were removed because they contained missing or infinite "
            "values in the columns used by the backend."
        )

    if r2_value is not None and np.isfinite(r2_value):
        if r2_value >= 0.95:
            st.success("The formula has a strong fit on the rows evaluated by the current backend.")
        elif r2_value >= 0.70:
            st.info("The formula has a moderate fit. Try a higher search effort or inspect the data for noise.")
        else:
            st.warning(
                "The fit is limited. Consider checking the target/features, data quality, noise, "
                "and search effort."
            )

    st.caption(
        "Important: the existing backend calculates metrics on the cleaned dataset used by its workflow. "
        "These are not independent test-set metrics; a separate train/test workflow would require backend changes."
    )

    # Download a report using only values returned by the existing backend.
    report = {
        "target": result.get("target"),
        "formula": result.get("formula"),
        "raw_formula": result.get("raw_formula"),
        "input_variables": result.get("variables", []),
        "search_effort": st.session_state.get("formula_effort", effort),
        "mse": result.get("mse"),
        "mae": result.get("mae"),
        "r2": result.get("r2"),
        "complexity_nodes": result.get("complexity"),
        "rows_used": result.get("n_rows"),
        "rows_dropped": result.get("n_dropped"),
        "unique_expressions_explored": result.get("n_explored"),
        "note": "Metrics are calculated by the current backend and are not independent test-set metrics.",
    }
    st.markdown("### 5. Export results")
    report_csv = pd.DataFrame([{
        **{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in report.items()}
    }]).to_csv(index=False).encode("utf-8")
    report_json = json.dumps(report, indent=2, default=str).encode("utf-8")

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Download formula report (CSV)",
            data=report_csv,
            file_name="formula_discovery_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "⬇️ Download formula report (JSON)",
            data=report_json,
            file_name="formula_discovery_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("Technical result details"):
        st.json(report)

st.markdown("---")
st.markdown(
    "<p class='subtle' style='text-align:center'>AI Formula Discovery · Genetic Programming / Symbolic Regression</p>",
    unsafe_allow_html=True,
)