import streamlit as st
from itables.streamlit import interactive_table
from app.components.dataset_loader import load_dataset

st.set_page_config(page_title="error-demo", layout="wide")

st.title("error-demo")
st.caption("Tabular Error Injection, Detection & Cleaning Demo")

df = load_dataset()

if df is not None:
    st.markdown("## 2. Dataset Preview")
    interactive_table(df, paging=True, lengthMenu=[10, 25, 50, 100])
else:
    st.info("Upload a CSV or load the built-in dataset to begin.")