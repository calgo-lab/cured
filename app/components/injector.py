import streamlit as st
from tab_err.api.high_level import create_errors_with_config
from tab_err.error_mechanism import ECAR, ENAR, EAR
from tab_err.error_type import WrongUnit, Typo

def highlight_errors(df, mask):
    # map each element of the mask to a CSS string
    css = mask.map(lambda v: "background-color: #ff6b6b" if v == 1 else "")
    return df.style.apply(lambda _: css, axis=None)



def inject_errors_ui(df):
    st.markdown("## Inject Errors With tab-err")
    description = """This part of the demo allows the user to use error models of the form: (mechanism, type, rate) to perturb the data using tab-err (https://pypi.org/project/tab-err/).
    To perturb the data, select an error mechanism, type, and rate and click `Inject errors`. To revert to undo the errors, click `Revert to original dataset`.
    """
    
    st.markdown(
        description
    )

    # Initialize original dataset
    if "original_dataset" not in st.session_state:
        st.session_state.original_dataset = df.copy()

    # # Initialize session_state keys safely
    # if "dataset" not in st.session_state:
    #     st.session_state.dataset = df.copy()
    # if "error_mask" not in st.session_state:
    #     st.session_state.error_mask = None



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
        df_copy = st.session_state.test_df.copy()        
        
        # tab-err injection logic
        perturbed_df, error_mask, config = create_errors_with_config(df_copy, error_rate=error_rate, error_mechanisms_to_include=[EAR(), ECAR()], error_types_to_include=[WrongUnit({"wrong_unit_scaling": lambda x: x / 1e6}), Typo()], seed=1)

        st.session_state.dataset = perturbed_df
        st.session_state.error_mask = error_mask
        st.session_state.perturbation_config = config
        if error_mask.any().any():
            st.success(f"Injected errors at {error_rate * 100}%.")
        else:
            st.error("Choose a larger error rate.")

    if revert_button:
        st.session_state.dataset = st.session_state.test_df.copy()
        st.session_state.error_mask = None
        st.success("Reverted to original dataset.")
        revert_button = None

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
