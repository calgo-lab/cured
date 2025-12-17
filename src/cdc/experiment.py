from __future__ import annotations

import random
from logging import getLogger
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tab_err import ErrorMechanism
from tab_err.api.high_level import create_errors
from tab_err.error_mechanism import EAR, ECAR, ENAR
from tab_err.error_type import MissingValue, Mistype

from .cleaner import ConformalForestCleaner
from .cleaner._base import BaseCleaner
from .config import DATA_PATH, N_JOBS, RESULTS_PATH
from .data import TaskType, guess_task_type, split_columns_into_categorical_and_numerical

target_column_name = "target"
logger = getLogger(__name__)


class Experiment:
    def __init__(self, name: str, dataset_id: int, train_size: float = 0.8, data_path: Path = DATA_PATH, seed: int | None = None):
        self.name: str = name
        self.dataset_id: int = dataset_id
        self._train_size: float = train_size
        self._seed: int | None = seed
        self._data: pd.DataFrame = pd.read_csv(data_path / f"{self.dataset_id}.csv")
        self._categorical_columns, self._numerical_columns = split_columns_into_categorical_and_numerical(self._data.drop(columns=target_column_name))

        train, test = train_test_split(self._data, train_size=train_size, random_state=self._seed)
        self.task_type: TaskType = guess_task_type(self._data[target_column_name])

        # NOTE: We copy here to make sure we decouple them
        self.X_train: pd.DataFrame = train.drop(columns=target_column_name).copy()
        self.y_train: pd.Series = train[target_column_name].copy()

        self.X_test: pd.DataFrame = test.drop(columns=target_column_name).copy()
        self.y_test: pd.Series = test[target_column_name].copy()

        self.X_perturbed: pd.DataFrame | None = None
        self.error_mask: pd.DataFrame | None = None
        self.error_rate: float | None = None
        self.error_mechanism: type[ErrorMechanism] | None = None

        self.X_cleaned: pd.DataFrame | None = None
        self.cleaned_mask: pd.DataFrame | None = None
        self.X_multiple_cleaned: pd.DataFrame | None = None
        self.multiple_cleaned_mask: pd.DataFrame | None = None

        self._model: Pipeline | None = None
        self._cleaner: BaseCleaner | None = None
        self.confidence_level: float | None = None

    def fit_and_get_baseline_model(self) -> Pipeline:
        if self._model is None:
            feature_transformation = ColumnTransformer(
                transformers=[
                    ("categorical_features", OneHotEncoder(handle_unknown="ignore"), self._categorical_columns),
                    ("scaled_numeric", StandardScaler(), self._numerical_columns),
                ],
                sparse_threshold=0,
            )

            if self.task_type == TaskType.CLASSIFICATION:
                msg = f"Fitting RandomForestClassifier for dataset {self.dataset_id}"
                logger.debug(msg)

                predictor = RandomForestClassifier(random_state=self._seed, n_jobs=N_JOBS)

            elif self.task_type == TaskType.REGRESSION:
                msg = f"Fitting RandomForestRegressor for dataset {self.dataset_id}"
                logger.debug(msg)

                predictor = RandomForestRegressor(random_state=self._seed, n_jobs=N_JOBS)

            else:
                msg = "Only Regression or Classification tasks are supported."
                raise ValueError(msg)

            self._model = Pipeline([("preprocess", feature_transformation), ("predictor", predictor)]).fit(X=self.X_train, y=self.y_train)
        return self._model

    def perturb_and_get_perturbed_test_data_and_error_mask(self, error_rate: float, error_mechanism: type[ErrorMechanism]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if type(self.X_perturbed) is not pd.DataFrame or type(self.error_mask) is not pd.DataFrame:
            if isinstance(error_mechanism, type) and not issubclass(error_mechanism, ErrorMechanism):
                msg = "'error_mechanism' must be subclasses of ErrorMechanism"
                raise ValueError(msg)

            self.error_rate = error_rate
            self.error_mechanism = error_mechanism
            self.X_perturbed, self.error_mask = create_errors(
                data=self.X_test,
                error_rate=self.error_rate,
                error_types_to_exclude=[Mistype(), MissingValue()],
                error_mechanisms_to_exclude=[mechanism() for mechanism in [ECAR, EAR, ENAR] if mechanism != self.error_mechanism],
                seed=self._seed,
            )

        return self.X_perturbed, self.error_mask

    def fit_and_get_cleaner(self, confidence_level: float) -> BaseCleaner:
        if self._cleaner is None:
            self.confidence_level = confidence_level
            self._cleaner = ConformalForestCleaner(confidence_level=self.confidence_level, seed=self._seed).fit(
                data=self.X_train, sk_params={"random_state": self._seed, "n_jobs": N_JOBS}
            )

        return self._cleaner

    def clean_and_get_cleaned_test_data_and_cleaned_mask(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if type(self.X_perturbed) is not pd.DataFrame:
            msg = "Run 'perturb_and_get_perturbed_test_data_and_error_mask' first."
            raise RuntimeError(msg)

        if not isinstance(self._cleaner, BaseCleaner):
            msg = "Run 'fit_and_get_cleaner' first."
            raise TypeError(msg)

        if type(self.X_cleaned) is not pd.DataFrame or type(self.cleaned_mask) is not pd.DataFrame:
            self.X_cleaned, self.cleaned_mask = self._cleaner.transform(self.X_perturbed)

        return self.X_cleaned, self.cleaned_mask

    def multiple_clean_and_get_multiple_cleaned_test_data_and_multiple_cleaned_mask(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if type(self.X_perturbed) is not pd.DataFrame:
            msg = "Run 'perturb_and_get_perturbed_test_data_and_error_mask' first."
            raise RuntimeError(msg)

        if not isinstance(self._cleaner, BaseCleaner):
            msg = "Run 'fit_and_get_cleaner' first."
            raise TypeError(msg)

        if type(self.X_multiple_cleaned) is not pd.DataFrame or type(self.multiple_cleaned_mask) is not pd.DataFrame:
            self.X_multiple_cleaned, self.multiple_cleaned_mask = self._cleaner.multiple_transform(self.X_perturbed)

        return self.X_multiple_cleaned, self.multiple_cleaned_mask

    def calculate_and_get_metrics(self) -> dict:
        if type(self.X_perturbed) is not pd.DataFrame or type(self.error_mask) is not pd.DataFrame:
            msg = "Run 'perturb_and_get_perturbed_test_data' first."
            raise RuntimeError(msg)

        if type(self.X_cleaned) is not pd.DataFrame or type(self.cleaned_mask) is not pd.DataFrame:
            msg = "Run 'clean_and_get_cleaned_test_data_and_cleaned_mask' first."
            raise RuntimeError(msg)

        if type(self.X_multiple_cleaned) is not pd.DataFrame or type(self.multiple_cleaned_mask) is not pd.DataFrame:
            msg = "Run 'multiple_clean_and_get_multiple_cleaned_test_data_and_multiple_cleaned_mask' first."
            raise RuntimeError(msg)

        model = self.fit_and_get_baseline_model()
        y_hat = model.predict(self.X_test)
        y_hat_perturbed = model.predict(self.X_perturbed)
        y_hat_cleaned = model.predict(self.X_cleaned)
        y_hat_multiple_cleaned = model.predict(self.X_multiple_cleaned)

        number_of_errors = self.error_mask.sum().sum()
        cleaned_mask = self.X_cleaned != self.X_perturbed
        multiple_cleaned_mask = self.X_multiple_cleaned != self.X_perturbed

        error_detection_tpr = (self.error_mask & cleaned_mask).sum().sum() / number_of_errors
        error_detection_fpr = (~self.error_mask & cleaned_mask).sum().sum() / (~self.error_mask).sum().sum()
        multiple_error_detection_tpr = (self.error_mask & multiple_cleaned_mask).sum().sum() / number_of_errors
        multiple_error_detection_fpr = (~self.error_mask & multiple_cleaned_mask).sum().sum() / (~self.error_mask).sum().sum()
        metrics = {
            "error_detection_tpr": error_detection_tpr,
            "error_detection_fpr": error_detection_fpr,
            "multiple_error_detection_tpr": multiple_error_detection_tpr,
            "multiple_error_detection_fpr": multiple_error_detection_fpr,
        }

        if self.task_type is TaskType.CLASSIFICATION:
            metrics.update(
                {
                    "orig_1": f1_score(self.y_test, y_hat, average="micro"),
                    "orig_2": f1_score(self.y_test, y_hat, average="macro"),
                    "orig_3": f1_score(self.y_test, y_hat, average="weighted"),
                    "perturbed_1": f1_score(self.y_test, y_hat_perturbed, average="micro"),
                    "perturbed_2": f1_score(self.y_test, y_hat_perturbed, average="macro"),
                    "perturbed_3": f1_score(self.y_test, y_hat_perturbed, average="weighted"),
                    "cleaned_1": f1_score(self.y_test, y_hat_cleaned, average="micro"),
                    "cleaned_2": f1_score(self.y_test, y_hat_cleaned, average="macro"),
                    "cleaned_3": f1_score(self.y_test, y_hat_cleaned, average="weighted"),
                    "multiple_cleaned_1": f1_score(self.y_test, y_hat_multiple_cleaned, average="micro"),
                    "multiple_cleaned_2": f1_score(self.y_test, y_hat_multiple_cleaned, average="macro"),
                    "multiple_cleaned_3": f1_score(self.y_test, y_hat_multiple_cleaned, average="weighted"),
                }
            )

        elif self.task_type is TaskType.REGRESSION:
            metrics.update(
                {
                    "orig_1": r2_score(self.y_test, y_hat),
                    "orig_2": mean_absolute_error(self.y_test, y_hat),
                    "orig_3": mean_squared_error(self.y_test, y_hat),
                    "perturbed_1": r2_score(self.y_test, y_hat_perturbed),
                    "perturbed_2": mean_absolute_error(self.y_test, y_hat_perturbed),
                    "perturbed_3": mean_squared_error(self.y_test, y_hat_perturbed),
                    "cleaned_1": r2_score(self.y_test, y_hat_cleaned),
                    "cleaned_2": mean_absolute_error(self.y_test, y_hat_cleaned),
                    "cleaned_3": mean_squared_error(self.y_test, y_hat_cleaned),
                    "multiple_cleaned_1": r2_score(self.y_test, y_hat_multiple_cleaned),
                    "multiple_cleaned_2": mean_absolute_error(self.y_test, y_hat_multiple_cleaned),
                    "multiple_cleaned_3": mean_squared_error(self.y_test, y_hat_multiple_cleaned),
                }
            )

        else:
            msg = "Currently, only Classification or Regression Experiments are supported."
            raise RuntimeError(msg)

        return metrics

    def compute_and_get_results(self) -> pd.Series:
        if self.error_mechanism is None:
            msg = "Run 'perturb_and_get_perturbed_test_data' first."
            raise RuntimeError(msg)

        results = {
            "experiment_name": self.name,
            "dataset_id": self.dataset_id,
            "task_type": self.task_type,
            "error_rate": self.error_rate,
            "error_mechanism": self.error_mechanism.__name__,
            "confidence_level": self.confidence_level,
            "train_size": self._train_size,
            "seed": f"{self._seed}-{random.randint(0, 0xFFFF):04X}" if self._seed is None else self._seed,  # just a random string so we can save the results
        }
        results.update(self.calculate_and_get_metrics())

        return pd.Series(results)

    def compute_and_save_results(self, results_path: Path = RESULTS_PATH) -> None:
        if type(results_path) is not Path:
            results_path = Path(results_path)

        results = self.compute_and_get_results()
        results_dir_path: Path = (
            results_path
            / str(results["experiment_name"])
            / str(results["dataset_id"])
            / str(results["train_size"])
            / str(results["error_mechanism"])
            / str(results["error_rate"])
            / str(results["confidence_level"])
            / str(results["seed"])
        )
        results_dir_path.mkdir(parents=True, exist_ok=True)

        results.to_csv(
            results_dir_path / "results.csv",
        )
