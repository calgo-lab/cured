# components/conformal_cleaner.py
import pandas as pd
import streamlit as st
import numpy as np
from conformal_data_cleaning.demo_interface import fit_and_get_cleaner
from conformal_data_cleaning.cleaner import ConformalForestCleaner

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
    seed:int = 41
):
    """

    Input: 
        the perturbed dataframe given in the demo
        the confidence level


    Output:
        Cleaned dataframe
    """

    fit_cleaner = fit_and_get_cleaner(cleaner=ConformalForestCleaner.__name__, train_df=train_df, confidence_level=c_level, seed=seed)  # Autogluon goes here
    cleaned_test_df, cleaned_mask = fit_cleaner.transform(test_df)

    return cleaned_test_df, cleaned_mask


### Streamlit logic

def conformal_cleaning_ui():
    st.markdown("## Conformal Data Cleaning")
    st.markdown(
        "Clean injected errors using a conformal predictor with coverage guarantees, this uses the library (Link? there is no PyPi). "
        "The confidence level affects the coverage of the conformal predictor, with higher confidence levels being more conservative; a value of 0.9 or higher is recommended."

    )

    if "cleaned_dataset" not in st.session_state:
        st.session_state.cleaned_dataset = None
    if "clean_mask" not in st.session_state:
        st.session_state.clean_mask = None

    c_level = st.slider(
        "Confidence Level", min_value=0.5, max_value=0.9999, value=.99, step=0.001
    )

    col1, col2 = st.columns(2)
    code_str = """
    from conformal_data_cleaning.cleaner import ConformalForestCleaner

    cleaner = ConformalForestCleaner(train_df, confidence_level)
    
    cleaned_test_df, cleaned_mask = cleaner.transform(test_df)
    """
    
    with col1:
        clean_button = st.button("Run conformal cleaning")
    with col2:
        with st.expander("Code Example"):
            st.code(code_str, language="python")

    if clean_button:
        with st.spinner("Running conformal cleaner..."):
            cleaned_df, mask = conformal_clean(
                st.session_state.dataset,
                st.session_state.train_df,
                c_level=c_level,
            )

        st.session_state.cleaned_dataset = cleaned_df
        st.session_state.clean_mask = mask


    # === Visualization ===
    if st.session_state.cleaned_dataset is not None:
        error_detection_tpr = (st.session_state.error_mask & st.session_state.clean_mask).sum().sum() / st.session_state.error_mask.sum().sum()
        error_detection_fpr = (~st.session_state.error_mask & st.session_state.clean_mask).sum().sum() / (~st.session_state.error_mask).sum().sum()


        # --- METRICS ROW ---
        m1, m2, m3 = st.columns([1, 1, 2])

        with m1:
            st.metric("TPR - True Positive Rate", f"{error_detection_tpr:.2%}")

        with m2:
            st.metric("FPR - False Positive Rate", f"{error_detection_fpr:.2%}")

        # --- LEGEND + FORMULA CARD ---
        with m3:
            st.markdown(
                """
                <div style="
                    background-color: var(--secondary-background-color);
                    padding:14px;
                    border-radius:10px;
                    border:1px solid var(--border-color, rgba(0,0,0,0.1));
                ">

                <div style="font-size:14px; font-weight:600; margin-bottom:8px;">
                    Legend
                </div>

                <div style="margin-bottom:10px;">
                    <span style="background-color:#ff6b6b;color:white;padding:4px 8px;border-radius:6px;font-size:12px;">
                        Error only
                    </span>
                    &nbsp;
                    <span style="background-color:#3498db;color:white;padding:4px 8px;border-radius:6px;font-size:12px;">
                        CDC Modified
                    </span>
                    &nbsp;
                    <span style="background-color:#03fc24;color:black;padding:4px 8px;border-radius:6px;font-size:12px;">
                        Error & CDC Modified
                    </span>
                </div>

                <div style="font-size:14px; font-weight:600; margin-bottom:6px;">
                    Detection Formula
                </div>

                <div style="
                    font-family:monospace;
                    background-color: rgba(0,0,0,0.03);
                    padding:10px;
                    border-radius:6px;
                    font-size:13px;
                ">
                    TPR = 
                    <span style="color:#03fc24;">TP</span> / 
                    (<span style="color:#03fc24;">TP</span> + 
                    <span style="color:#ff6b6b;">FN</span>)
                    <br>
                    FPR = 
                    <span style="color:#3498db;">FP</span> / 
                    (<span style="color:#3498db;">FP</span> + 
                    <span style="color:#9ca3af;">TN</span>)
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown("### Cleaned dataset")
        styled_df = highlight_errors(
            st.session_state.cleaned_dataset,
            st.session_state.error_mask,
            st.session_state.clean_mask
        )
        st.dataframe(styled_df)

        

    else:
        st.info("Run the cleaner to view cleaned data.")
