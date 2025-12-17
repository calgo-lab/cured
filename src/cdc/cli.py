from enum import Enum
from logging import getLogger
from typing import Annotated

import typer
from tab_err.error_mechanism import EAR, ECAR, ENAR
from typer.params import Option

from .experiment import Experiment

cli = typer.Typer()

logger = getLogger(__name__)


class ErrorMechanismArgument(str, Enum):
    ECAR = "ECAR"
    EAR = "EAR"
    ENAR = "ENAR"


@cli.command()
def main(
    experiment_name: Annotated[str, Option(...)],
    dataset_id: Annotated[int, Option(...)],
    error_rate: Annotated[float, Option(min=0.0001, max=0.99999)],
    error_mechanism: Annotated[ErrorMechanismArgument, Option(...)],
    num_repetitions: Annotated[int, Option(...)],
    confidence_level: Annotated[float, Option(min=0.0001, max=0.99999)],
    train_size: Annotated[float, Option(min=0.0001, max=0.99999)] = 0.8,
):
    if error_mechanism == ErrorMechanismArgument.ECAR:
        error_mechanism_ = ECAR

    if error_mechanism == ErrorMechanismArgument.ENAR:
        error_mechanism_ = ENAR

    if error_mechanism == ErrorMechanismArgument.EAR:
        error_mechanism_ = EAR

    for seed in range(num_repetitions):
        msg = f"Starting repetition {seed}"
        logger.info(msg)

        experiment = Experiment(name=experiment_name, dataset_id=dataset_id, seed=seed, train_size=train_size)

        _ = experiment.fit_and_get_baseline_model()
        _ = experiment.perturb_and_get_perturbed_test_data_and_error_mask(error_rate=error_rate, error_mechanism=error_mechanism_)
        _ = experiment.fit_and_get_cleaner(confidence_level=confidence_level)
        _ = experiment.clean_and_get_cleaned_test_data_and_cleaned_mask()
        _ = experiment.multiple_clean_and_get_multiple_cleaned_test_data_and_multiple_cleaned_mask()

        experiment.compute_and_save_results()
