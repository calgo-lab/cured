import pandas as pd
from cdc.cleaner._base import BaseCleaner
from cdc.cleaner import ConformalForestCleaner

# Vars


def fit_and_get_cleaner(X_train:pd.DataFrame, confidence_level: float, njobs: int, seed: int = 42) -> BaseCleaner:
    cleaner = ConformalForestCleaner(confidence_level=confidence_level, seed=seed).fit(
        data=X_train, sk_params={"random_state": seed, "n_jobs": njobs}
    )
    return cleaner


def multiple_clean_and_get_multiple_cleaned_test_data_and_multiple_cleaned_mask(X_perturbed, cleaner) -> tuple[pd.DataFrame, pd.DataFrame]:
    if type(X_perturbed) is not pd.DataFrame:
        msg = "Run 'perturb_and_get_perturbed_test_data_and_error_mask' first."
        raise RuntimeError(msg)

    if not isinstance(cleaner, BaseCleaner):
        msg = "Run 'fit_and_get_cleaner' first."
        raise TypeError(msg)

    X_multiple_cleaned, multiple_cleaned_mask = cleaner.multiple_transform(X_perturbed)

    return X_multiple_cleaned, multiple_cleaned_mask
