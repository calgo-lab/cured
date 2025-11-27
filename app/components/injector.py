import streamlit as st
from itables.streamlit import interactive_table
from tab_err.api.high_level import create_errors

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
    ) / 100

    # Buttons side by side
    col1, col2 = st.columns(2)
    with col1:
        inject_button = st.button("Inject errors")
    with col2:
        revert_button = st.button("Revert to original dataset")

    if inject_button:
        df_copy = st.session_state.dataset.copy()
        # TODO: Replace with tab-err injection logic
        perturbed_df, error_mask = create_errors(df_copy, error_rate=error_rate)

        st.session_state.dataset = perturbed_df
        st.session_state.error_mask = error_mask
        st.success(f"Injected errors at {error_rate * 100}%.")

    if revert_button:
        st.session_state.dataset = st.session_state.original_dataset.copy()
        st.session_state.error_mask = None
        st.success("Reverted to original dataset.")

    # Display dataset and error mask side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Data")
        interactive_table(st.session_state.dataset, paging=True, lengthMenu=[10, 25, 50, 100])

    with col2:
        st.markdown("### Error Mask")
        if st.session_state.get("error_mask") is not None:
            error_mask_numeric = st.session_state.error_mask.astype(int)
            interactive_table(error_mask_numeric, paging=True, lengthMenu=[10, 25, 50, 100])
        else:
            st.info("No error mask yet.")
