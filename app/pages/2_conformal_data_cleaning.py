import streamlit as st
from app.components.conformal_cleaner import conformal_cleaning_ui

st.set_page_config(page_title="Conformal Cleaning", layout="wide")

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("No dataset loaded. Go back to the main page to load one.")
else:
    conformal_cleaning_ui(st.session_state.dataset)
