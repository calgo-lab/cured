import streamlit as st
from tab_err.api.high_level import create_errors_with_config
from tab_err.error_mechanism import ECAR, ENAR, EAR
from tab_err.error_type import WrongUnit, Typo, Outlier, AddDelta, Extraneous


ERROR_MECH_DESCRIPTIONS = {
        "ECAR": "Locations of errors are not dependent on the data in the table.",
        "EAR": "Locations of errors are dependent on other columns in the table.",
        "ENAR": "Locations of errors are dependent on the column in which they occur.",
    }

ERROR_TYPE_DESCRIPTIONS = {
        "Wrong Unit Scale": "Cells are scaled by a constant factor.",
        "Add Delta": "A constant value is added to cells.",
        "Outlier": "Outliers of the column's distribution replace cells.",
        "Typo": "Characters in the string are replaced by adjacent characters from the qwerty keyboard.",
        "Extraneous": "Random characters are appended or prepended to the string.",
    }


def highlight_errors(df, mask):
    # map each element of the mask to a CSS string
    css = mask.map(lambda v: "background-color: #ff6b6b" if v == 1 else "")
    return df.style.apply(lambda _: css, axis=None)



def inject_errors_ui(df):
    st.markdown("## Inject Errors With tab-err")
    description = """This part of the demo allows the user to use error models of the form: (mechanism, type, rate) to perturb the data using tab-err (https://pypi.org/project/tab-err/). 
    Any error mechanisms or types listed will be used to perturb the table.
    To perturb the data, select an error mechanism, type, and rate and click `Inject errors`. 
    To revert to undo the errors, click `Revert to original dataset`.
    To see a description of the various error mechanisms, click `Error mechanism descriptions`.
    To see a description of the various error types, click `Error type descriptions`.
    Certain error types apply only to numeric and others only to categorical columns.
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

    # --- ERROR MECHANISM SELECTION ---
    valid_error_mechs = {
        "ECAR": ECAR(),
        "EAR": EAR(),
        "ENAR": ENAR(),
    }

    selected_mech_names = st.multiselect(
        "Error mechanisms",
        list(valid_error_mechs.keys()),
        default=["EAR"]
    )

    if not selected_mech_names:
        st.warning("Select at least one error mechanism. Defaulting to EAR.")
        selected_mech_names = ["EAR"]

    selected_mechs = [valid_error_mechs[n] for n in selected_mech_names]



    # --- ERROR TYPE SELECTION ---
    valid_error_types = {
        "Wrong Unit": WrongUnit({"wrong_unit_scaling": lambda x: x / 1e6}),
        "Add Delta": AddDelta(),
        "Outlier": Outlier(),
        "Typo": Typo(),
        "Extraneous": Extraneous(),
    }

    selected_type_names = st.multiselect(
        "Error types",
        list(valid_error_types.keys()),
        default=["Wrong Unit", "Typo"]
    )

    if not selected_type_names:
        st.warning("Select at least one error type. Defaulting to Wrong Unit and Typo")
        selected_type_names = ["Wrong Unit", "Typo"]

    selected_types = [valid_error_types[n] for n in selected_type_names]



    # --- ERROR RATE SELECTION ---
    error_rate = st.slider(
        "Error rate (% of cells)", min_value=0, max_value=100, value=10, step=1
    ) /100

    # --- CODE STRING ---
    code_str = "perturbed_df, error_mask = create_errors(df, error_rate=error_rate)"

    # Buttons side by side
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        inject_button = st.button("Inject errors")
    with col2:
        revert_button = st.button("Revert to original dataset")
    with col3:
        with st.popover("Error mechanism descriptions"):
            for name, desc in ERROR_MECH_DESCRIPTIONS.items():
                st.markdown(f"**{name}**")
                st.caption(desc)
    with col4:
        with st.popover("Error type descriptions"):
            for name, desc in ERROR_TYPE_DESCRIPTIONS.items():
                st.markdown(f"**{name}**")
                st.caption(desc)
    with col5:
        with st.popover("tab-err code"):
            st.markdown(f"```python\n{code_str}\n```")

    if inject_button:
        df_copy = st.session_state.test_df.copy()        
        
        # tab-err injection logic
        perturbed_df, error_mask, config = create_errors_with_config(df_copy, error_rate=error_rate, error_mechanisms_to_include=selected_mechs, error_types_to_include=selected_types, seed=1)

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
