import streamlit as st
from app.components.dataset_loader import load_dataset
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Error Demo", layout="centered", initial_sidebar_state="expanded")

st.title("Generating, Cleaning, and Characterizing Realistic Errors in Tables")
st.caption("Tabular Error Generation, Cleaning, & Characterization Demo")

description = """This demo shows how tabular data can be errored, cleaned, and characterized. It brings together the results of three works on error generation, conformal data cleaning, and error mechanism detection which are explored in detail in the following papers:
1. [Towards Realistic Error Models for Tabular Data](https://doi.org/10.1145/3774914)
2. [From Data Imputation to Data Cleaning - Automated Cleaning of Tabular Data Improves Downstream Predictive Performance](https://proceedings.mlr.press/v238/jager24a.html)
3. [MechDetect: Detecting Data-Dependent Errors](https://arxiv.org/abs/2512.04138)

The user can proceed by uploading a dataset (or using the built-in dataset), perturbing the dataset with tab-err, cleaning the dataset with conformal data cleaning, and finally characterizing the distribution of errors using mech detect.
"""

st.markdown("## Description")
st.markdown(description)
