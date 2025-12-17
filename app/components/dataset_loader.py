import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import random

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
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        train, test = _split_and_store(df)
        st.success("Dataset loaded from file.")

    if load_dummy:
        df = make_dummy_dataset()
        train, test = _split_and_store(df)
        st.success("Dummy dataset loaded.")

    return test


def make_dummy_dataset(n=100):
    base_age = [22, 35, 58, 41, 19, 63]
    base_income = [35000, 72000, 54000, 61000, 29000, 83000]
    base_city = ["Berlin", "Paris", "Berlin", "Rome", "Madrid", "Vienna"]
    base_children = [False, True, True, False, False, True]

    ages = [random.choice(base_age) + random.randint(-2, 2) for _ in range(n)]
    incomes = [random.choice(base_income) + random.randint(-5000, 5000) for _ in range(n)]
    cities = [random.choice(base_city) for _ in range(n)]
    has_children = [random.choice(base_children) for _ in range(n)]

    print(len(has_children))
    return pd.DataFrame({
        "age": ages,
        "income": incomes,
        "city": cities,
        "has_children": has_children
    })
