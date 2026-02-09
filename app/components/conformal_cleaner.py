# components/conformal_cleaner.py
import pandas as pd
import streamlit as st
import numpy as np
from conformal_data_cleaning.demo_interface import fit_and_get_cleaner
from conformal_data_cleaning.cleaner import ConformalForestCleaner
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import root_mean_squared_error, f1_score

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
            return "background-color: #008b74"  # teal - cleaned and there was an error
        elif error_locations:
            return "background-color: #ff6b6b"  # red - error
        elif cleaning_locations:
            return "background-color: #85c1e9"  # blue - cleaned but no error
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
    with st.expander("Description", expanded=True):
        st.markdown(
            "Clean injected errors using a conformal predictor with coverage guarantees and evaluate cleaning with a downstream ML task as in [this paper](https://proceedings.mlr.press/v238/jager24a.html). "
            "The confidence level affects the coverage of the conformal predictor, with higher confidence levels being more conservative; a value of 0.9 or higher is recommended. "
            "Additionally, a comparison of model performance on the original data, the perturbed data, and the cleaned data is conducted."
        )

    if "cleaned_dataset" not in st.session_state:
        st.session_state.cleaned_dataset = None
    if "clean_mask" not in st.session_state:
        st.session_state.clean_mask = None

    c_level = st.slider(
        "Confidence Level", min_value=0.5, max_value=0.9999, value=.99, step=0.001
    )

    col1, col2, col3 = st.columns(3)
    code_str = """
    from conformal_data_cleaning.cleaner import ConformalForestCleaner

    cleaner = ConformalForestCleaner(train_df, confidence_level)
    
    cleaned_test_df, cleaned_mask = cleaner.transform(test_df)
    """
    
    with col1:
        clean_button = st.button("Run Conformal Cleaning")
    with col2:
        task_type = st.radio(
            "Select Task Type:",
            options=["Classification", "Regression"],
            horizontal=True,
            help="Determines the target variable type and evaluation metrics.",
            index=1
        )
    with col3:
        with st.expander("Code Example"):
            st.code(code_str, language="python")

    if clean_button:
        with st.spinner("Running conformal cleaner..."):
            

            # --- Clean data ---
            cleaner_train_dataset = st.session_state.train_df.drop(columns=["target"])  # Only clean the features
            cleaner_train_target = st.session_state.train_df["target"]
            perturbed_dataset = st.session_state.dataset


            cleaned_df, mask = conformal_clean(
                perturbed_dataset,
                cleaner_train_dataset,
                c_level=c_level,
            )


            # --- Downstream task ---
            # Train model
            cat_feats = cleaner_train_dataset.select_dtypes(include=["object", "category"]).columns.tolist()
            if task_type == "Classification":
                model = CatBoostClassifier(iterations = 200, depth = 4, verbose = 0)
            else:
                model = CatBoostRegressor(iterations = 200, depth = 4, verbose = 0)
            
            model.fit(cleaner_train_dataset, cleaner_train_target, cat_features=cat_feats)

            # Evaluate model
            results = []
            evaluation_y = st.session_state.test_df["target"]
            unaltered_X = st.session_state.test_df.drop(columns=["target"])

            test_datasets = {
                "Original Data": (unaltered_X, evaluation_y),
                "Perturbed Data": (perturbed_dataset, evaluation_y),
                "Cleaned Data": (cleaned_df, evaluation_y)
            }

            for name, (X_t, y_t) in test_datasets.items():
                preds = model.predict(X_t)

                if task_type == "Classification":
                    metric_name = "F1-Score"
                    metric_val = f1_score(y_t, preds, average="weighted")
                else:  # Regression
                    metric_name = "RMSE"
                    metric_val = root_mean_squared_error(y_t, preds)
                results.append({"Dataset": name, metric_name: metric_val})

            st.session_state.ml_task_summary = pd.DataFrame(results)

        st.session_state.cleaned_dataset = cleaned_df
        st.session_state.clean_mask = mask


    # === Visualization ===
    if st.session_state.cleaned_dataset is not None:
        error_detection_tpr = (st.session_state.error_mask & st.session_state.clean_mask).sum().sum() / st.session_state.error_mask.sum().sum()
        error_detection_fpr = (~st.session_state.error_mask & st.session_state.clean_mask).sum().sum() / (~st.session_state.error_mask).sum().sum()


        # --- BLOCK 1: Detection Metrics & Legend ---
        # Split: Left (Detection Stats) | Right (Legend)
        col_detection, col_legend = st.columns([1, 1])

        with col_detection:
            st.subheader("Error Detection Stats")
            # Create a 2-column layout strictly for TPR/FPR
            d1, d2 = st.columns(2)
            with d1:
                st.metric("TPR (Sensitivity)", f"{error_detection_tpr:.2%}", help="True Positive Rate: Percentage of actual errors correctly flagged.")
            with d2:
                st.metric("FPR (Fall-out)", f"{error_detection_fpr:.2%}", help="False Positive Rate: Percentage of clean data incorrectly flagged as error.")

        with col_legend:
            # Your existing HTML Legend
            st.markdown(
                """
                <div style="
                    background-color: var(--secondary-background-color);
                    padding:14px;
                    border-radius:10px;
                    border:1px solid var(--border-color, rgba(0,0,0,0.1));
                    height: 100%;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                
                <div style="font-size:14px; font-weight:600; margin-bottom:8px;">Legend</div>

                <div style="margin-bottom:10px;">
                    <span style="background-color:#ff6b6b;color:white;padding:4px 8px;border-radius:6px;font-size:12px;">Error only</span>
                    &nbsp;
                    <span style="background-color:#3498db;color:white;padding:4px 8px;border-radius:6px;font-size:12px;">CDC Modified</span>
                    &nbsp;
                    <span style="background-color:#008b74;color:white;padding:4px 8px;border-radius:6px;font-size:12px;">Error and Modified</span>
                </div>

                <div style="font-size:14px; font-weight:600; margin-bottom:6px;">Detection Formula</div>

                <div style="font-family:monospace; background-color: rgba(0,0,0,0.03); padding:10px; border-radius:6px; font-size:13px;">
                    TPR = <span style="color:#03fc2008b744;">TP</span> / (<span style="color:#008b74;">TP</span> + <span style="color:#ff6b6b;">FN</span>)<br>
                    FPR = <span style="color:#3498db;">FP</span> / (<span style="color:#3498db;">FP</span> + <span style="color:#9ca3af;">TN</span>)
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        # --- BLOCK 2: Downstream ML Performance ---
        # Determine metric name dynamically for the header
        metric_key = "RMSE" if "RMSE" in st.session_state.ml_task_summary.columns else "F1-Score"
        
        # Explicit Text Header
        st.markdown(f"#### Downstream Model Performance: {metric_key}")
        st.caption(f"Evaluation of a catboost model on the original, perturbed, and cleaned test sets.")

        # Create 3 columns for the 3 datasets
        p1, p2, p3 = st.columns(3)
        cols = [p1, p2, p3]

        # Helper to safe-get data
        def get_metric(idx):
            return st.session_state.ml_task_summary.iloc[idx] if idx < len(st.session_state.ml_task_summary) else None

        for i, col in enumerate(cols):
            row_data = get_metric(i)
            with col:
                if row_data is not None:
                    st.metric(
                        label=row_data["Dataset"], 
                        value=f"{row_data[metric_key]:.4f}",
                        delta_color="off" # "off" keeps it neutral, or use "normal"/"inverse" if you have a baseline
                    )

        st.divider()

        # --- BLOCK 3: Cleaned Data Preview ---
        st.markdown("### Cleaned Data Preview")
        styled_df = highlight_errors(
            st.session_state.cleaned_dataset,
            st.session_state.error_mask,
            st.session_state.clean_mask
        )
        st.dataframe(styled_df)
    
    else:
        st.info("Run the cleaner to view cleaned data.")
