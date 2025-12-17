import streamlit as st
from app.components.injector import inject_errors_ui

st.set_page_config(page_title="Inject Errors", layout="wide")

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("No dataset loaded. Go back to the main page to load one.")
else:
    inject_errors_ui(st.session_state.test_df)
