# components/conformal_cleaner.py
import pandas as pd
import streamlit as st
from conformal_data_cleaning.cleaner.autogluon import ConformalAutoGluonCleaner

def highlight_errors(df, error_mask, clean_mask):
    """
    df: DataFrame
    error_mask: same shape as df, 1 for error, 0 else
    clean_mask: same shape as df, 1 for cleaned, 0 else
    Coloring:
        Red    -> error only
        Green  -> clean only
        Blue   -> both error and clean
    """
    def color_cell(val, row, col):
        error_locations = error_mask.iloc[row, col]
        cleaning_locations = clean_mask.iloc[row, col]
        if error_locations and cleaning_locations:
            return "background-color: #03fc24"  # green - cleaned and there was an error
        elif error_locations:
            return "background-color: #ff6b6b"  # red - error
        elif cleaning_locations:
            return "background-color: #3498db"  # blue - cleaned but no error
        else:
            return ""
    
    # Vectorized via applymap with row/col index
    def styler_func(x):
        return pd.DataFrame(
            [[color_cell(x.iloc[i, j], i, j) for j in range(x.shape[1])] for i in range(x.shape[0])],
            index=x.index,
            columns=x.columns
        )
    
    return df.style.apply(styler_func, axis=None)



def conformal_clean(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    c_level: float,
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
    model_hps = {"hyperparameters": {"RF": {}}}
    cleaner: ConformalAutoGluonCleaner = ConformalAutoGluonCleaner(confidence_level=c_level, seed = seed)
    fit_cleaner = cleaner.fit(train_df, ci_ag_fit_params=model_hps)
    cleaned_test_df, cleaned_mask = fit_cleaner.transform(test_df)

    return cleaned_test_df, cleaned_mask


### Streamlit logic

def conformal_cleaning_ui():
    st.markdown("## Conformal Data Cleaning")
    st.markdown(
        "Clean injected errors using a conformal predictor with "
        "coverage guarantees."
    )

    if "cleaned_dataset" not in st.session_state:
        st.session_state.cleaned_dataset = None
    if "clean_mask" not in st.session_state:
        st.session_state.clean_mask = None

    c_level = st.slider(
        "Confidence Level", min_value=0.0001, max_value=0.9999, value=.95, step=0.001
    )

    clean_button = st.button("Run conformal cleaning")

    if clean_button:
        with st.spinner("Running conformal cleaner..."):
            cleaned_df, mask = conformal_clean(
                st.session_state.test_df,
                st.session_state.train_df,
                c_level=c_level,
            )

        st.session_state.cleaned_dataset = cleaned_df
        st.session_state.clean_mask = mask

        st.success("Cleaning completed.")

    # === Visualization ===
    if st.session_state.cleaned_dataset is not None:
        st.markdown("### Cleaned dataset")
        styled_df = highlight_errors(
            st.session_state.cleaned_dataset,
            st.session_state.error_mask,
            st.session_state.clean_mask
        )
        st.dataframe(styled_df)


    else:
        st.info("Run the cleaner to view cleaned data.")
