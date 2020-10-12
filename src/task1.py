#!/usr/bin/env python
"""The entry point for the scripts for Task 1."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from typing_extensions import Final

from typings import CSVData, CSVHeader
from utilities.data import (
    create_submission_file,
    print_array,
    print_array_statistics,
    read_csv,
    visualize_data,
)

TASK_DATA_DIRECTORY: Final[str] = "data/task1"
TRAINING_DATA_NAME: Final[str] = "X_train.csv"
TRAINING_LABELS_NAME: Final[str] = "y_train.csv"
TEST_DATA_PATH: Final[str] = "X_test.csv"


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
    X_train_w_outliers = imputer.fit_transform(X_train)

    # Use LOF for outlier detection
    outliers = LocalOutlierFactor(contamination=0.09).fit_predict(X_train_w_outliers)

    # Take out the outliers
    X_train = X_train[outliers == 1]
    Y_train = Y_train[outliers == 1]

    # (Re-)impute the data without the outliers
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = xgb.XGBRegressor()

    if args.mode == "eval":
        # Returns an array of the k cross-validation R^2 scores
        scores = cross_val_score(model, X_train, Y_train, cv=args.cross_val, scoring="r2")
        avg_score = np.mean(scores)
        print(f"Average R^2 score is: {avg_score:.4f}")
    elif args.mode == "final":
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        submission: Any = np.stack([test_ids, Y_pred], 1)  # Add IDs
        create_submission_file(args.output, submission, header=("id", "y"))
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


def __data_diagnostics(data: CSVData, labels: CSVData, header: CSVHeader) -> None:
    # Print preview of data
    print_array(data, header)
    print_array(labels, header)

    # Print statistics of data
    print_array_statistics(data)
    print_array_statistics(labels)

    # Tensorboard will stuck on "computing PCA" if there are non-number values in the array
    data = np.nan_to_num(data)

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
        help="the path by which to save the output CSV (only used in the 'final' mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["eval", "final"],
        default="eval",
        help="whether to evaluate using cross-validation or do final training to generate output",
    )
    parser.add_argument(
        "-k",
        "--cross-val",
        type=int,
        default=10,
        help="the k for k-fold cross-validation",
    )
    parser.add_argument("--diagnose", action="store_true", help="enable data diagnostics")
    __main(parser.parse_args())
