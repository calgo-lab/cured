import streamlit as st
from app.components.dataset_loader import load_dataset

st.set_page_config(page_title="Load Data", layout="centered", initial_sidebar_state="expanded")

load_dataset()

if st.session_state.test_df is not None:
    st.markdown("## Dataset Preview")
    st.dataframe(st.session_state.test_df)
else:
    st.info("Upload a CSV or load the built-in dataset to begin.")
