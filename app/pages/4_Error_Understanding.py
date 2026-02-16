import streamlit as st
from app.components.mech_detect import mech_detect_ui

st.set_page_config(page_title="MechDetect", layout="centered", initial_sidebar_state="expanded")

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("No dataset loaded. Go back to the main page to load one.")
elif "error_mask" not in st.session_state or st.session_state.error_mask is None:
    st.warning("No errors to characterize. Use the error injection page to generate some.")
else:
    mech_detect_ui()
