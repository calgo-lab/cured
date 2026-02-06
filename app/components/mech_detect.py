# app/components/mech_detect.py
import streamlit as st
import pandas as pd
import numpy as np
from mechdetect import MechDetector

# Import your MechDetect class when ready
# from mechdetect import MechDetector

def highlight_errors(df: pd.DataFrame, mask: pd.DataFrame):
    """
    Highlight errored cells in a dataframe using a CSS style.
    """
    css = mask.map(lambda v: "background-color: #ff6b6b" if v == 1 else "")
    return df.style.apply(lambda _: css, axis=None)

def mech_detect_ui():
    """
    Streamlit UI for running MechDetect on a selected column of a dataframe,
    including a live preview of the dataset with optional error highlighting.
    """
    if "dataset" not in st.session_state or st.session_state.dataset is None:
        st.warning("No dataset loaded. Go back to the main page to load one.")
        return

    if "detected_mech" not in st.session_state:
        st.session_state.detected_mech = None
    
    if "detected_column" not in st.session_state:
        st.session_state.detected_column = None

    df = st.session_state.dataset

    st.header("MechDetect: Error Mechanism Detection")
    
    description = """MechDetect trains machine learning models to predict the error mask of a column given various subsets of the data, 
    the performance of these models is compared using statistical tests to detect mechanism as shown [here](https://github.com/calgo-lab/MechDetect/blob/MechDetect%2B%2B/src/mechdetect.py). 
    The column used to detect the error is selected, and after running MechDetect, the classified error mechanism is provided along with the actual error mechanism and the two p-values from MechDetect's internal tests (for more details, see the [paper](https://arxiv.org/abs/2512.04138)).
    """
    st.markdown(description)

    # Let the user select a column
    columns = [col for col in df.columns if col != "target"]
    column = st.selectbox("Select column to detect error mechanism", columns)

    col1, col2 = st.columns(2)

    # --- Run MechDetect button ---
    with col1:
        if st.button("Run MechDetect"):
            if df[column].isnull().all():
                st.error("Selected column has only missing values.")
            else:
                with st.spinner(f"Detecting error mechanism in '{column}'..."):
                    try:
                        detector = MechDetector(alpha=0.05, cv_folds=5, seed=42, n_jobs=1)
                        detected_mech, p1, p2 = detector.detect(
                            st.session_state.clean_test_df.drop(columns=["target"]),
                            st.session_state.error_mask,
                            column=column
                        )
                        # Save results in session state
                        st.session_state.detected_mech = detected_mech
                        st.session_state.p1 = p1
                        st.session_state.p2 = p2
                        st.session_state.detected_column = column
                        
 

                    except Exception as e:
                        st.error(f"Error running MechDetect: {e}")

    # --- MechDetect Code Example ---
    mechdetect_code = """
    from mechdetect import MechDetector

    detector = MechDetector(alpha=0.05, cv_folds=5)
    detected_mech, p1, p2 = detector.detect(test_df, error_mask, column)
    """
    if "p1" in st.session_state and st.session_state.p1 is not None and np.isnan(st.session_state.p1):
        st.error("Dataset is too small to run MechDetect")
        st.stop()

    with col2:
        with st.expander("Code Example"):
            st.code(mechdetect_code, language="python")

    # --- Display Results if available ---
    if st.session_state.detected_mech is not None and st.session_state.detected_column is not None:
        st.subheader("Results")
        mech_col_1, mech_col_2 = st.columns(2)
        with mech_col_1:
            st.markdown(f"**Detected mechanism:** `{st.session_state.detected_mech}`")
        with mech_col_2:
            st.markdown(f"**Injected mechanism:** `{st.session_state.perturbation_config.columns[st.session_state.detected_column][0].error_mechanism.__class__.__name__}`")

        col1_res, col2_res = st.columns(2)
        col1_res.metric("p1", f"{st.session_state.p1:.4f}")
        if st.session_state.p2 is not None:
            col2_res.metric("p2", f"{st.session_state.p2:.4f}")
        else:
            col2_res.markdown("**p2:** N/A")

    # --- Show dataset preview ---
    if "error_mask" in st.session_state and st.session_state.error_mask is not None:
        styled = highlight_errors(df, st.session_state.error_mask)
        st.dataframe(styled)
    else:
        st.dataframe(df)