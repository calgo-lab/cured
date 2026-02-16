import streamlit as st
from app.components.dataset_loader import load_dataset
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(page_title="error-demo", layout="wide")

st.title("error-demo")
st.caption("Tabular Error Injection, Detection & Cleaning Demo")

st.markdown("## 0. About")
st.markdown("This demo shows how tabular data can be errored, cleaned, and characterized.")

load_dataset()

if st.session_state.test_df is not None:
    st.markdown("## 2. Dataset Preview")
    st.dataframe(st.session_state.test_df)
else:
    st.info("Upload a CSV or load the built-in dataset to begin.")