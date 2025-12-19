import streamlit as st
from app.components.mech_detect import mech_detect_ui

st.set_page_config(page_title="Mech Detect", layout="wide")

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("No dataset loaded. Go back to the main page to load one.")
else:
    mech_detect_ui()
