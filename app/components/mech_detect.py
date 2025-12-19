# app/components/mech_detect.py
import streamlit as st
import pandas as pd
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

    df = st.session_state.dataset

    st.header("MechDetect: Error Mechanism Detection")

    # Let the user select a column
    column = st.selectbox("Select column to detect error mechanism", df.columns)

    # User inputs for MechDetect parameters
    st.subheader("MechDetect Settings")
    alpha = st.slider(
        "Significance level (α)",
        min_value=0.01, max_value=0.3, value=0.1, step=0.01,
        help="The significance level for the tests in MechDetect"
    )
    n_cv = st.number_input(
        "Number of cross-validation iterations",
        min_value=1, max_value=50, value=5, step=1,
        help="Number of CV folds/iterations for MechDetect"
    )

    # Show dataset preview
    if "error_mask" in st.session_state and st.session_state.error_mask is not None:
        styled = highlight_errors(df, st.session_state.error_mask)
        st.dataframe(styled)
    else:
        st.dataframe(df)





    if st.button("Run MechDetect"):
        if df[column].isnull().all():
            st.error("Selected column has only missing values.")
            return

        with st.spinner(f"Detecting error mechanism in '{column}'..."):
            try:
                detector = MechDetector(alpha=alpha, cv_folds=n_cv, seed=42, n_jobs=1)
                detected_mech, p1, p2 = detector.detect(
                    st.session_state.clean_test_df,
                    st.session_state.error_mask,
                    column=column
                )

            except Exception as e:
                st.error(f"Error running MechDetect: {e}")

            else:
                st.success("Detection completed!")
                st.subheader("Results")

                # Display a clean summary
                st.markdown(f"**Detected mechanism:** `{detected_mech}`")

                col1, col2 = st.columns(2)
                col1.metric("p1", f"{p1:.4f}")
                
                # Only show p2 if it's not None
                if p2 is not None:
                    col2.metric("p2", f"{p2:.4f}")
                else:
                    col2.markdown("**p2:** N/A")

                # Optional: also show a simple dataframe summary
                summary_df = pd.DataFrame({
                    "Mechanism": [detected_mech],
                    "p1": [p1],
                    "p2": [p2 if p2 is not None else "N/A"]
                })
                st.table(summary_df)

