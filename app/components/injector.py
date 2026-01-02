import streamlit as st
from tab_err.api.high_level import create_errors_with_config

def highlight_errors(df, mask):
    # map each element of the mask to a CSS string
    css = mask.map(lambda v: "background-color: #ff6b6b" if v == 1 else "")
    return df.style.apply(lambda _: css, axis=None)



def inject_errors_ui(df):
    st.markdown("## Inject Errors")
    st.markdown(
        "Specify the error rate below and click **Inject**."
    )

    # Initialize original dataset
    if "original_dataset" not in st.session_state:
        st.session_state.original_dataset = df.copy()

    # Initialize session_state keys safely
    if "dataset" not in st.session_state:
        st.session_state.dataset = df.copy()
    if "error_mask" not in st.session_state:
        st.session_state.error_mask = None

    # User specifies error rate
    error_rate = st.slider(
        "Error rate (% of cells)", min_value=0, max_value=100, value=10, step=1
    ) /100

    # Buttons side by side
    col1, col2 = st.columns(2)
    with col1:
        inject_button = st.button("Inject errors")
    with col2:
        revert_button = st.button("Revert to original dataset")

    if inject_button:
        df_copy = st.session_state.original_dataset.copy()        
        
        # TODO: Replace with tab-err injection logic
        perturbed_df, error_mask, config = create_errors_with_config(df_copy, error_rate=error_rate)

        st.session_state.dataset = perturbed_df
        st.session_state.error_mask = error_mask
        st.session_state.perturbation_config = config
        if error_mask.any().any():
            st.success(f"Injected errors at {error_rate * 100}%.")
        else:
            st.error("Choose a larger error rate.")

    if revert_button:
        st.session_state.dataset = st.session_state.original_dataset.copy()
        st.session_state.error_mask = None
        st.success("Reverted to original dataset.")

    if "error_mask" in st.session_state and st.session_state.error_mask is not None:
        styled = highlight_errors(
            st.session_state.dataset,
            st.session_state.error_mask,
        )

        # MUST use st.dataframe for Styler support
        #st.dataframe(styled, width='stretch')
        
        st.dataframe(styled)
    else:
        st.dataframe(st.session_state.dataset)
