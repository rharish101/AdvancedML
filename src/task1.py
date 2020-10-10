#!/usr/bin/env python
"""The entry point for the scripts for Task 1."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Any, cast

import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from typing_extensions import Final

from typings import CSVData, CSVHeader
from utilities.data import (
    create_submission_file,
    print_array,
    print_array_statistics,
    read_csv,
    visualize_data,
)

TASK_DATA_DIRECTORY: Final = "data/task1"
TRAINING_DATA_NAME: Final = "X_train.csv"
TRAINING_LABELS_NAME: Final = "y_train.csv"
TEST_DATA_PATH: Final = "X_test.csv"


def __main(args: Namespace) -> None:
    # Read in data
    X_train, X_header = read_csv(f"{args.data_dir}/{TRAINING_DATA_NAME}")
    Y_train, _ = read_csv(f"{args.data_dir}/{TRAINING_LABELS_NAME}")
    X_test, _ = read_csv(f"{args.data_dir}/{TEST_DATA_PATH}")

    if X_train is None or Y_train is None or X_test is None:
        raise RuntimeError("There was a problem with reading CSV data")

    if args.diagnose:
        __data_diagnostics(X_train, Y_train, header=X_header or ())

    # Remove training IDs, as they are in sorted order for training data
    X_train = X_train[:, 1:]
    Y_train = Y_train[:, 1:]

    # Save test IDs as we need to add them to the submission file
    test_ids = X_test[:, 0]
    X_test = X_test[:, 1:]

    # We can substitute this for a more complex imputer later on
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = xgb.XGBRegressor()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Add IDs
    submission = np.stack([test_ids, Y_pred], 1)
    create_submission_file(args.output, submission, header=("id", "y"))


def __data_diagnostics(data: CSVData, labels: CSVData, header: CSVHeader) -> None:
    # Print preview of data
    print_array(data, header)
    print_array(labels, header)

    # Print statistics of data
    print_array_statistics(data)
    print_array_statistics(labels)

    # Tensorboard will stuck on "computing PCA" if there are non-number values in the array
    data = cast(Any, np).nan_to_num(data)

    # Create a TensorBoard projector to visualize data
    visualize_data(data[:, 1:], data[:, 0].astype(int), "input_data")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="The entry point for the scripts for Task 1",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/task1",
        help="path to the directory containing the task data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dist/submission1.csv",
        help="the path by which to save the output CSV",
    )
    parser.add_argument("--diagnose", action="store_true", help="enable data diagnostics")
    __main(parser.parse_args())
