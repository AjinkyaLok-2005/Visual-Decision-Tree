# app.py
import streamlit as st
import pandas as pd
import numpy as np
from tree import build_tree_trace, predict_single, tree_to_json
from viz import build_dot, dot_to_png_bytes
from utils import load_csv, df_overview, export_python_predict_function
import base64
import io
import json

st.set_page_config(page_title="Visual Decision Tree Builder", layout="wide")

st.title("🔍 Visual Interactive Decision Tree Builder")

# Sidebar
st.sidebar.header("Data & Settings")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_example = st.sidebar.selectbox("Or use example dataset", ["--", "Iris (example)"])
impurity = st.sidebar.selectbox("Impurity", ["gini", "entropy"])
max_depth = st.sidebar.number_input("Max depth (0 = no limit)", min_value=0, value=0, step=1)
min_samples_split = st.sidebar.number_input("Min samples to split", min_value=2, value=2, step=1)
if uploaded:
    df = load_csv(uploaded)
elif use_example == "Iris (example)":
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
else:
    st.info("Upload a CSV or choose an example to begin.")
    st.stop()

st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.sidebar.markdown("---")

# Dataset overview
st.header("1. Dataset Overview")
overview = df_overview(df)
st.write(f"Rows: {overview['n_rows']} — Columns: {overview['n_cols']}")
st.dataframe(df.head(5))
col = st.selectbox("Select target column (for classification)", df.columns.tolist())
feature_cols = [c for c in df.columns if c != col]

# Preprocess target to integer labels
y_raw = df[col]
if y_raw.dtype == object or y_raw.dtype.name == 'category':
    labels = {v:i for i,v in enumerate(y_raw.astype(str).unique())}
    y = y_raw.astype(str).map(labels).values
else:
    # make discrete labels if numeric but limited uniques
    if df[col].nunique() <= 10:
        labels = {v:i for i,v in enumerate(sorted(df[col].unique()))}
        y = df[col].map(labels).values
    else:
        # if too many unique continuous target — warn
        st.error("Target appears continuous. This demo focuses on classification tasks with discrete labels.")
        st.stop()

X_df = df[feature_cols].copy()

# Stepper / Build trace
st.header("2. Build Decision Tree (Step-by-step)")
if max_depth == 0:
    max_depth_val = None
else:
    max_depth_val = int(max_depth)

trace_generator = build_tree_trace(X_df, y, feature_cols, impurity=impurity, max_depth=max_depth_val, min_samples_split=min_samples_split)
snapshots = list(trace_generator)

# store snapshots in session_state for stepper persistence
if "snapshots" not in st.session_state:
    st.session_state.snapshots = snapshots
    st.session_state.step_index = 0
    st.session_state.score = 0
    st.session_state.total_guesses = 0

snapshots = st.session_state.snapshots
n_steps = len(snapshots)
st.write(f"Total splits available: {n_steps}")

col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Controls")
    step = st.number_input("Step index", min_value=0, max_value=max(0,n_steps-1), value=st.session_state.step_index, step=1)
    if st.button("Previous"):
        st.session_state.step_index = max(0, st.session_state.step_index - 1)
    if st.button("Next"):
        st.session_state.step_index = min(n_steps-1, st.session_state.step_index + 1)
    st.session_state.step_index = step

    st.checkbox("Gamification Mode", key="gamemode")
    if st.session_state.gamemode:
        st.write("Make your prediction for the next split feature:")
        current_idx = st.session_state.step_index
        if current_idx < n_steps:
            # show list of candidate features (from all_steps metadata)
            # For simplicity show all features
            candidate = st.radio("Choose feature", feature_cols)
            if st.button("Submit Guess"):
                st.session_state.total_guesses += 1
                actual = snapshots[current_idx]["step"]["chosen_feature"]
                if candidate == actual:
                    st.success("Correct!")
                    st.session_state.score += 1
                else:
                    st.error(f"Wrong — actual: {actual}")
                st.write(f"Score: {st.session_state.score} / {st.session_state.total_guesses}")

with col2:
    st.subheader("Current Split Details")
    idx = st.session_state.step_index
    snap = snapshots[idx]
    step_info = snap["step"]
    if step_info is None:
        st.write("No splits (pure or too small).")
    else:
        st.markdown(f"**Step {idx+1}** — depth {step_info['depth']} — feature **{step_info['chosen_feature']}** <= **{step_info['threshold']}**")
        st.write("Split info:")
        st.json(step_info["split_info"])
        st.write("Class counts at node:")
        st.write(dict(step_info["class_counts"]))

# Visualization
st.header("3. Tree Visualization")
current_snap = snapshots[st.session_state.step_index]
root = current_snap["root"]
active = current_snap["step"]
dot_src = build_dot(root, active_step=active, show_counts=True)
st.graphviz_chart(dot_src, use_container_width=True)

# Export buttons
st.header("4. Exports")
colA, colB, colC = st.columns(3)
with colA:
    st.download_button("Download JSON", data=tree_to_json(root), file_name="tree.json", mime="application/json")
with colB:
    code = export_python_predict_function(root)
    st.download_button("Download Python predict()", data=code, file_name="predict_tree.py", mime="text/x-python")
with colC:
    # PNG export
    try:
        png = dot_to_png_bytes(dot_src)
        st.download_button("Download PNG", data=png, file_name="tree.png", mime="image/png")
    except Exception as e:
        st.error(f"PNG export failed: {e}. Ensure Graphviz system binary is installed.")

st.markdown("---")
st.write("Tips: Use Iris or smaller datasets for smooth step-by-step interactions. For big datasets the step count can be large.")
