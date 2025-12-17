# components/conformal_cleaner.py
import pandas as pd
import streamlit as st
from conformal_data_cleaning.cleaner.autogluon import ConformalAutoGluonCleaner


def conformal_clean(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    alpha: float,
    njobs: int = 4,
    seed:int = 42
):
    """

    Input: 
        the perturbed dataframe given in the demo
        the alpha value for conformal coverage


    Output:
        Cleaned dataframe
    """
    cleaner: ConformalAutoGluonCleaner = ConformalAutoGluonCleaner(confidence_level= 1-alpha, seed = seed)
    fit_cleaner = cleaner.fit(train_df)
    cleaned_test_df, cleaned_mask = fit_cleaner.transform(test_df)

    return cleaned_test_df, cleaned_mask


### Streamlit logic

def conformal_cleaning_ui(df):
    st.markdown("## Conformal Data Cleaning")
    st.markdown(
        "Clean injected errors using a conformal predictor with "
        "coverage guarantees."
    )

    if "cleaned_dataset" not in st.session_state:
        st.session_state.cleaned_dataset = None
    if "clean_mask" not in st.session_state:
        st.session_state.clean_mask = None

    alpha = st.selectbox(
        "Miscoverage level (α)",
        options=[0.01, 0.05, 0.1],
        index=2  # default value, e.g., 0.9
    )

    clean_button = st.button("Run conformal cleaning")

    if clean_button:
        with st.spinner("Running conformal cleaner..."):
            cleaned_df, mask = conformal_clean(
                st.session_state.test_df,
                st.session_state.train_df,
                alpha=alpha,
            )

        st.session_state.cleaned_dataset = cleaned_df
        st.session_state.clean_mask = mask

        st.success("Cleaning completed.")

    # === Visualization ===
    if st.session_state.cleaned_dataset is not None:
        st.markdown("### Cleaned dataset")
        st.dataframe(st.session_state.cleaned_dataset)

    else:
        st.info("Run the cleaner to view cleaned data.")
