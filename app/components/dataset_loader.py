import streamlit as st
import pandas as pd
import hashlib
from sklearn.model_selection import train_test_split
import copy

def _file_hash(file) -> str:
    """
    Stable hash for detecting new uploads
    """
    return hashlib.md5(file.getvalue()).hexdigest()

def _reset_dataset_state():
    """
    Clears all session_state keys that depend on the dataset
    """
    keys_to_clear = [
        # Core dataset
        "dataset",
        "train_df",
        "test_df",
        "clean_test_df",
        "original_dataset",

        # Error injection
        "error_mask",
        "perturbation_config",

        # Conformal cleaning
        "cleaned_dataset",
        "clean_mask",
    ]

    for k in keys_to_clear:
        st.session_state.pop(k, None)


def _split_and_store(df, train_frac=0.8, seed=42):
    train_df, test_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        shuffle=True,
    )

    st.session_state.dataset = df
    st.session_state.train_df = train_df.reset_index(drop=True)
    st.session_state.test_df = test_df.reset_index(drop=True)
    st.session_state.clean_test_df = copy.deepcopy(test_df)
    return train_df, test_df


def load_dataset():
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "train_df" not in st.session_state:
        st.session_state.train_df = None
    if "test_df" not in st.session_state:
        st.session_state.test_df = None

    st.markdown("## 1. Load Dataset")
    st.markdown(
        "Upload your own CSV/Excel file or start with an example dataset. "
        "**80%** of the data will be used to train the cleaner; "
        "**20%** will be held out."
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
    test = None

    if uploaded is not None:
        current_hash = _file_hash(uploaded)

        if st.session_state.get("dataset_hash") != current_hash:
            _reset_dataset_state()
            st.session_state.dataset_hash = current_hash

            df = (
                pd.read_csv(uploaded)
                if uploaded.name.endswith(".csv")
                else pd.read_excel(uploaded)
            )

            train, test = _split_and_store(df)
            st.success("New dataset loaded and session state reset.")


    if load_dummy:
        dummy_hash = "DUMMY_DATASET"

        if st.session_state.get("dataset_hash") != dummy_hash:
            _reset_dataset_state()
            st.session_state.dataset_hash = dummy_hash

            df = load_dummy_dataset()
            train, test = _split_and_store(df)
            st.success("Dummy dataset loaded and session state reset.")

    return test


def load_dummy_dataset():
    return pd.read_csv("data/44969.csv") # .sample(200)
