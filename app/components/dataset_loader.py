import streamlit as st
import pandas as pd
import hashlib
from sklearn.model_selection import train_test_split
import copy
import re


# Hard limits based on container resources
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
MAX_ROWS = 10_000
MAX_COLS = 50

def load_data_securely(uploaded_file):
    if uploaded_file is None:
        return None

    # 1. Size Check (Prevent Disk/RAM DoS before parsing)
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File size exceeds 10MB limit.")
        st.stop()

    try:
        if uploaded_file.name.endswith('.csv'):
            # 2. Limit Columns + Rows + Engine
            # Explicitly use 'c' engine for speed or 'python' for stability
            df = pd.read_csv(
                uploaded_file, 
                nrows=MAX_ROWS, 
                usecols=lambda x: x not in range(MAX_COLS, 1000), # Cap wide-column attacks
            )
                
        elif uploaded_file.name.endswith('.xlsx'):
            # 3. XLSX specific: openpyxl + row/col limits
            # Note: Pandas doesn't support nrows with read_excel directly in all versions
            df = pd.read_excel(uploaded_file, engine='openpyxl').iloc[:MAX_ROWS, :MAX_COLS]
            
        else:
            st.error("Unsupported file type.")
            st.stop()

        # 4. Sanitization (Hardened Regex for CSV Injection)
        # Prevents leading characters that trigger Excel/Sheets execution
        if not df.empty:
            trigger_pattern = re.compile(r"^[=\+\-\@\t\r]")
            cols = df.select_dtypes(include=['object']).columns
            for col in cols:
                df[col] = df[col].astype(str).apply(
                    lambda x: f"'{x}" if trigger_pattern.match(x) else x
                )

        return df

    except Exception as e:
        # 5. Generic error to prevent info leakage (paths, versions, etc.)
        st.error(f"Unable to parse file. Examine the number of columns and rows. {e}")
        st.stop()

def _file_hash(file) -> str:
    """
    Stable hash for detecting new uploads
    """
    return hashlib.md5(file.getvalue()).hexdigest()

def _reset_dataset_state():
    """
    Clears all session_state keys that depend on the dataset
    """
    keys_to_clear = [
        # Core dataset
        "dataset",
        "train_df",
        "test_df",
        "clean_test_df",
        "original_dataset",
        "train_target_col",
        "test_target_col",

        # Error injection
        "error_mask",
        "perturbation_config",

        # Conformal cleaning
        "cleaned_dataset",
        "clean_mask",
    ]

    for k in keys_to_clear:
        st.session_state.pop(k, None)

def _split_and_store(df, train_frac=0.8, seed=42):
    train_df, test_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        shuffle=True,
    )

    st.session_state.dataset = df
    st.session_state.train_df = train_df.reset_index(drop=True)
    st.session_state.test_df = test_df.reset_index(drop=True)
    st.session_state.clean_test_df = copy.deepcopy(test_df)
    st.session_state.train_target_col = train_df["target"]
    st.session_state.test_target_col = test_df["target"]
    return train_df, test_df


def load_dataset():
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "train_df" not in st.session_state:
        st.session_state.train_df = None
    if "test_df" not in st.session_state:
        st.session_state.test_df = None

    st.markdown("## Load Dataset")

    st.markdown(
        "Upload your own CSV/Excel file or start with an example dataset. "
        "**80%** of the data will be used to train the cleaner; "
        "**20%** will be held out."
    )

    st.markdown("The dataset is limited to 10,000 rows and 10 columns including a column named `target` for use in the downstream task of the data cleaning section.")

    st.markdown("---")

    st.markdown("### Option A - Upload a file")
    uploaded = st.file_uploader(  # Can configure size using server.maxUploadSize
        "Supports CSV and Excel (.xlsx) files",
        type=["csv", "xlsx"],
        key="uploader"
    )

    st.markdown("### Option B - Use an example dataset")
    st.caption(
        "Loads a dataset with the OpenML ID: 44969 subsampled to 8 columns."
    )

    load_dummy = st.button("Load dummy dataset", use_container_width=True)

    test = None

    if uploaded is not None:
        # --- RESET FILE POINTER ---
        current_hash = _file_hash(uploaded)

        if st.session_state.get("dataset_hash") != current_hash:
            _reset_dataset_state()
            st.session_state.dataset_hash = current_hash

            df = load_data_securely(uploaded)

            # --- ROW SUBSAMPLE ---
            if len(df) > MAX_ROWS:
                df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

            # --- COLUMN SUBSAMPLE ---
            cols = list(df.columns)

            if "target" not in cols:
                st.error(
                    "Your dataset must contain a column named **'target'**.\n\n"
                    "Please rename your label column to **target** and re-upload the file so the full demo works."
                )
                st.stop()

            # Always preserve target if present
            keep = ["target"]

            # Other columns (excluding target)
            others = [c for c in cols if c != "target"]

            # Subsample remaining columns if needed
            if len(keep) + len(others) > MAX_COLS:
                needed = MAX_COLS - len(keep)
                others = others[:needed]  # deterministic; use .sample(...) if you want randomness

            df = df[keep + others]

            int_cols = df.select_dtypes(include="int").columns
            df[int_cols] = df[int_cols].astype("float")

            train, test = _split_and_store(df)
            st.success("Dataset loaded.")


    if load_dummy:
        dummy_hash = "DUMMY_DATASET"

        if st.session_state.get("dataset_hash") != dummy_hash:
            _reset_dataset_state()
            st.session_state.dataset_hash = dummy_hash

            df = load_dummy_dataset()
            int_cols = df.select_dtypes(include="int").columns
            df[int_cols] = df[int_cols].astype("float")
            train, test = _split_and_store(df)
            st.success("Dummy dataset loaded and session state reset.")

    return test


def load_dummy_dataset():
    dummy_data = pd.read_csv("data/44969.csv").drop(columns=["ship_speed", "gas_generator_rate_of_revolutions", "hp_turbine_exit_pressure", "gt_compressor_outlet_air_pressure", "gt_compressor_outlet_air_temperature", "gas_turbine_exhaust_gas_pressure", "hp_turbine_exit_temperature"]) # .sample(200)
    
    # Reorder columns
    cols = ["target"] + [c for c in dummy_data.columns if c != "target"]
    dummy_data = dummy_data[cols]
    return dummy_data
