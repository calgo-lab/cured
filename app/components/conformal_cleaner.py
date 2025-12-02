import streamlit as st
import pandas as pd
import io
import requests

BACKEND_URL = "http://conformal-backend:8000/clean"  # replace with actual pod URL

def conformal_cleaning_ui(df: pd.DataFrame):
    st.header("Conformal Data Cleaning (Backend)")

    st.write("Preview of your dataset:")
    st.dataframe(df.head())

    # Select columns to clean
    columns = st.multiselect(
        "Select columns for cleaning",
        df.columns.tolist(),
        default=df.columns.tolist()
    )

    # Confidence level slider
    confidence_level = st.slider(
        "Confidence level",
        min_value=0.5,
        max_value=0.999,
        value=0.9,
        step=0.001
    )

    if st.button("Clean Data via Backend"):
        if not columns:
            st.warning("Please select at least one column.")
            return

        # Prepare dataset subset
        subset_df = df[columns]

        # Convert to CSV bytes
        csv_bytes = subset_df.to_csv(index=False).encode()

        try:
            with st.spinner("Sending data to conformal cleaning backend..."):
                response = requests.post(
                    BACKEND_URL,
                    files={"file": ("dataset.csv", csv_bytes)},
                    data={"confidence_level": confidence_level}
                )
                response.raise_for_status()

                # Read cleaned CSV
                cleaned_df = pd.read_csv(io.BytesIO(response.content))

                st.success(f"Data cleaned with {confidence_level*100:.1f}% confidence!")
                st.dataframe(cleaned_df.head())

                # Save to session state
                st.session_state.cleaned_dataset = cleaned_df

        except requests.RequestException as e:
            st.error(f"Error connecting to backend: {e}")
