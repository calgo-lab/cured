import streamlit as st
from app.components.conformal_cleaner import conformal_cleaning_ui

st.set_page_config(page_title="Conformal Data Cleaning", layout="centered", initial_sidebar_state="expanded")

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("No dataset loaded. Go back to the main page to load one.")
elif "error_mask" not in st.session_state or st.session_state.error_mask is None:
    st.warning("No errors to clean. Use the error injection page to generate some.")
else:
    conformal_cleaning_ui()
