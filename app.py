"""
app.py  -  Streamlit UI
------------------------
Run with:

    streamlit run app.py

The user only does 3 things:
    1. Upload a CSV file
    2. Pick the target (output) column
    3. Click "Discover Formula"

Everything else (Genetic Programming, simplification, metrics) runs in the
backend (backend.py) and only the FINAL formula is shown.
"""

import pandas as pd
import streamlit as st

from backend import discover_formula, EFFORT_PRESETS

st.set_page_config(page_title="AI Formula Discovery", page_icon="🧬", layout="centered")

st.title("🧬 AI Formula Discovery")
st.caption("Upload a CSV file and the AI will discover the mathematical formula hidden in your data "
           "using Genetic Programming (built from scratch).")

uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded is not None:
    try:
        uploaded.seek(0)          # app reruns on every click, so rewind the file first
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("Could not read this file. Please upload a valid CSV.")
        st.stop()

    if df.shape[1] < 2:
        st.error("The CSV needs at least 2 columns (inputs + one output column).")
        st.stop()

    st.write(f"Loaded **{len(df)} rows** and **{df.shape[1]} columns**.")
    st.dataframe(df.head(5))

    columns = list(df.columns)
    target = st.selectbox("Which column is the output (the one to find a formula for)?",
                          columns, index=len(columns) - 1)

    with st.expander("Advanced (optional)"):
        effort = st.select_slider(
            "Search effort (higher = slower but better for complex formulas)",
            options=list(EFFORT_PRESETS.keys()), value="Standard")

    if st.button("🔍 Discover Formula", type="primary"):
        try:
            with st.spinner("Evolving thousands of formulas... please wait"):
                result = discover_formula(df, target, effort)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception:
            st.error("Something went wrong while processing this data. "
                     "Please check that the CSV contains only numeric data.")
            st.stop()

        st.success("Formula discovered!")
        st.subheader("Discovered Formula")
        st.code(f"{result['target']} = {result['formula']}", language="text")

        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{result['r2']:.5f}")
        c2.metric("MAE", f"{result['mae']:.4f}")
        c3.metric("MSE", f"{result['mse']:.4f}")

        if result["r2"] < 0.95:
            st.info("The fit is not very tight. Try a higher search effort under "
                    "'Advanced', or the data may not follow a simple formula.")
else:
    st.info("👆 Upload a CSV to get started.")
