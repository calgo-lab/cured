import streamlit as st
import pandas as pd


def load_dataset():
    if "dataset" not in st.session_state:
        st.session_state.dataset = None

    st.markdown("## 1. Load Dataset")
    st.markdown(
        "Upload your own CSV/Excel file or start with an example dataset "
        "to explore error injection, detection, and cleaning."
    )

    st.markdown("---")

    st.markdown("### Option A — Upload a file")
    uploaded = st.file_uploader(
        "Supports CSV and Excel (.xlsx) files",
        type=["csv", "xlsx"],
        key="uploader"
    )

    st.markdown("### Option B — Use an example dataset")
    st.caption(
        "Loads a small synthetic dataset with numerical and categorical columns."
    )

    load_dummy = st.button("Load dummy dataset", use_container_width=True)

    if uploaded is not None:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.session_state.dataset = df
        st.success("Dataset loaded from file.")

    if load_dummy:
        st.session_state.dataset = make_dummy_dataset()
        st.success("Dummy dataset loaded.")

    return st.session_state.dataset


def make_dummy_dataset():
    return pd.DataFrame(
        {
            "age": [22, 35, 58, 41, 19, 63],
            "income": [35000, 72000, 54000, 61000, 29000, 83000],
            "city": ["Berlin", "Paris", "Berlin", "Rome", "Madrid", "Vienna"],
            "has_children": [False, True, True, False, False, True],
        }
    )
