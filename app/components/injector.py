import streamlit as st
from tab_err.api.high_level import create_errors_with_config
from tab_err.error_mechanism import ECAR, ENAR, EAR
from tab_err.error_type import WrongUnit, Typo, Outlier, AddDelta, Extraneous
import pandas as pd


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

def _reset_session_state():
    keys_to_clear = [
        # Error injection
        "error_mask",
        "perturbation_config",

        # Conformal cleaning
        "cleaned_dataset",
        "clean_mask",
        "ml_task_summary"

        # MechDetect
        "detected_mech",
        "p1",
        "p2",
        "detected_column"
    ]

    for k in keys_to_clear:
        st.session_state.pop(k, None)

def inject_errors_ui(df):
    st.markdown("## Inject Errors With tab-err")
    description = """This part of the demo allows the user to use error models of the form: (mechanism, type, rate) to perturb the features of the data using tab-err (https://pypi.org/project/tab-err/). 
    To perturb the data, select an error mechanism, type, and rate and click `Inject errors`. Perturbed cells will be highlighted red.
    To revert to undo the errors, click `Revert to original dataset`.
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

    # --- TOP BAR ---
    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        inject_button = st.button("Inject errors", use_container_width=True)

    with col2:
        revert_button = st.button("Revert dataset", use_container_width=True)

    # --- DROPDOWN CONTROL PANEL ---
    with col3:
        with st.popover("Error settings"):
            st.markdown("### Error Configuration")
            error_configuration_explanation = """
            Any error mechanisms or types listed will be used to perturb the table.
            To see a description of the various error mechanisms, click `Error mechanism descriptions`.
            To see a description of the various error types, click `Error type descriptions`.
            Certain error types apply only to numeric and others only to categorical columns.
            """
            st.caption(error_configuration_explanation)

            # -----------------------------
            # ERROR MECHANISM SELECTION
            # -----------------------------
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

            st.divider()

            # -----------------------------
            # ERROR TYPE SELECTION
            # -----------------------------
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

            st.divider()

            # -----------------------------
            # ERROR RATE
            # -----------------------------
            error_rate = (
                st.slider(
                    "Error rate (% of cells)",
                    min_value=0,
                    max_value=100,
                    value=10,
                    step=1
                ) / 100
            )

            st.divider()

            # -----------------------------
            # DESCRIPTIONS
            # -----------------------------
            with st.expander("Error mechanism descriptions"):
                for name, desc in ERROR_MECH_DESCRIPTIONS.items():
                    st.markdown(f"**{name}**")
                    st.caption(desc)

            with st.expander("Error type descriptions"):
                for name, desc in ERROR_TYPE_DESCRIPTIONS.items():
                    st.markdown(f"**{name}**")
                    st.caption(desc)

            st.divider()

            # -----------------------------
            # CODE PREVIEW
            # -----------------------------
            code_str = f"""
            from tab_err.api.high_level import create_errors

            perturbed_df, error_mask = create_errors(df, error_rate={error_rate})"""
            st.markdown("### Code Example")
            st.code(code_str, language="python")


    if inject_button:
        _reset_session_state()
        df_copy = st.session_state.test_df.copy()        
        df_copy = df_copy.drop(columns=["target"])
        # tab-err injection logic
        perturbed_df, error_mask, config = create_errors_with_config(df_copy, error_rate=error_rate, error_mechanisms_to_include=selected_mechs, error_types_to_include=selected_types, seed=1)

        st.session_state.dataset = perturbed_df # pd.concat([st.session_state.test_target_col, perturbed_df], axis=1)

        st.session_state.error_mask = error_mask # pd.concat([prepend_col, error_mask], axis=1)

        st.session_state.perturbation_config = config
        if error_mask.any().any():
            st.success(f"Injected errors at {error_rate * 100}%.")
        else:
            st.error("Choose a larger error rate.")

    if revert_button:
        st.session_state.dataset = st.session_state.test_df.drop(columns=["target"]).copy()
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
        st.dataframe(st.session_state.test_df.drop(columns=["target"]))
