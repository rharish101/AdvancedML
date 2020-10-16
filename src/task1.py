#!/usr/bin/env python
"""The entry point for the scripts for Task 1."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from datetime import datetime
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from typing_extensions import Final

from models.simple_nn import SimpleNN
from typings import BaseRegressor, CSVData, CSVHeader
from utilities.data import (
    create_submission_file,
    print_array,
    print_array_statistics,
    read_csv,
    visualize_data,
)
from utilities.dataset import DataSet
from utilities.nn import train_network

TASK_DATA_DIRECTORY: Final[str] = "data/task1"
TRAINING_DATA_NAME: Final[str] = "X_train.csv"
TRAINING_LABELS_NAME: Final[str] = "y_train.csv"
TEST_DATA_PATH: Final[str] = "X_test.csv"

torch.device("cuda:0")
tensorboard_writer = SummaryWriter(
    log_dir="logs/training" + datetime.now().strftime("-%Y%m%d-%H%M%S")
)


def __main(args: Namespace) -> None:
    # Read in data
    X_train, X_header = read_csv(f"{args.data_dir}/{TRAINING_DATA_NAME}")
    Y_train, _ = read_csv(f"{args.data_dir}/{TRAINING_LABELS_NAME}")

    if X_train is None or Y_train is None:
        raise RuntimeError("There was a problem with reading CSV data")

    if args.diagnose:
        __run_data_diagnostics(X_train, Y_train, header=X_header or ())

    X_train, Y_train, imputer, preserve = preprocess(X_train, Y_train)

    if args.pca:
        pca = PCA()
        X_train = pca.fit_transform(X_train)

    if args.model == "nn":
        # TODO: This should be removed after the NN model is complete
        __evaluate_nn_model(X_train, Y_train)
    else:
        model = __choose_model(args.model)

    if args.mode == "eval":
        score = evaluate_model(model, X_train, Y_train, k=args.cross_val)
        print(f"Average R^2 score is: {score:.4f}")

    elif args.mode == "final":
        X_test, _ = read_csv(f"{args.data_dir}/{TEST_DATA_PATH}")
        if X_test is None:
            raise RuntimeError("There was a problem with reading CSV data")

        # Save test IDs as we need to add them to the submission file
        test_ids = X_test[:, 0]
        X_test = X_test[:, 1:]

        X_test = imputer.transform(X_test)
        X_test = X_test[:, preserve]

        if args.pca:
            X_test = pca.transform(X_test)

        __finalise_model(model, X_train, Y_train, X_test, test_ids, args.output)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


def preprocess(
    X_train: CSVData, Y_train: CSVData
) -> Tuple[CSVData, CSVData, SimpleImputer, List[bool]]:
    """Preprocess the data.

    Parameters
    ----------
    X_train: The training data
    Y_train: The training labels

    Returns
    -------
    The preprocessed training data
    The preprocessed training labels
    The imputer for missing values
    The list of booleans indicating which features to preserve
    """
    # Remove training IDs, as they are in sorted order for training data
    X_train = X_train[:, 1:]
    Y_train = Y_train[:, 1:]

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

    preserve = __select_features_correlation(X_train, Y_train)
    X_train = X_train[:, preserve]

    return X_train, Y_train, imputer, preserve


def __log_to_tensorboard(env):
    for name, value in env.evaluation_result_list:
        tensorboard_writer.add_scalar(name, value, env.iteration)


def __choose_model(name: str) -> BaseRegressor:
    """Choose a model given the name."""
    if name == "xgb":
        return xgb.XGBRegressor()
    elif name == "nn":
        # TODO: This should ideally do:
        # return NNRegressor(param_1, param_2, ...)
        raise NotImplementedError(f"'{name}' model not implemented")
    else:
        raise ValueError(f"Invalid model name: {name}")


def evaluate_model(model: BaseRegressor, X_train: CSVData, Y_train: CSVData, k: int) -> float:
    """Perform cross-validation on the given dataset and return the R^2 score.

    Parameters
    ----------
    model: The regressor model
    X_train: The training data
    Y_train: The training labels
    k: The number of folds in k-fold cross-validation

    Returns
    -------
    The R^2 validation score
    """
    # Returns an array of the k cross-validation R^2 scores
    scores = cross_val_score(model, X_train, Y_train, cv=k, scoring="r2")
    avg_score = np.mean(scores)
    return avg_score


def __select_features_correlation(
    X_train, Y_train, minimum_target_correlation=0.001, maximum_mutual_correlation=0.90
):
    """Determine which features should be removed based on mutual/target correlation.

    Parameters
    ----------
    X_train: The training data
    Y_train: The corresponding labels
    minimum_target_correlation: Features with less corr with the target should be removed
    maximum_mutual_correlation: Features with more corr with other feature should be removed

    Returns
    -------
    A list indicating which feature should be preserved and which not
    """
    df = pd.concat([pd.DataFrame(X_train), pd.DataFrame(Y_train)], axis=1)

    cor = df.corr()

    cor_target = abs(cor.iloc[-1])[:-1]
    preserve = cor_target >= minimum_target_correlation

    # For every feature, see if there is another feauture with which it has high correlation
    for c in range(X_train.shape[1]):
        for f in range(c + 1, X_train.shape[1]):
            if cor.iloc[f, c] > maximum_mutual_correlation:
                preserve[c] = False
                break

    return preserve


def __finalise_model(
    model: BaseRegressor,
    X_train: CSVData,
    Y_train: CSVData,
    X_test: CSVData,
    test_ids: CSVData,
    output: str,
) -> None:
    """Train the model on the complete data and generate the submission file.

    Parameters
    ----------
    model: The regressor model
    X_train: The training data
    Y_train: The training labels
    X_test: The test data
    test_ids: The IDs for the test data
    """
    print()
    model.fit(X_train, Y_train, eval_set=[(X_train, Y_train)], callbacks=[__log_to_tensorboard])

    feature_importances = model.get_booster().get_score(importance_type="gain").items()

    feature_importances = sorted(feature_importances, key=lambda tuple: tuple[1], reverse=True)
    print("\n10 most important features:")
    print(feature_importances[:10])
    print("\n10 least important features:")
    print(feature_importances[:-11:-1])

    Y_pred = model.predict(X_test)
    submission: Any = np.stack([test_ids, Y_pred], 1)  # Add IDs
    create_submission_file(output, submission, header=("id", "y"))


def __evaluate_nn_model(X_train, Y_train):
    model = SimpleNN(X_train.shape[1])

    training_indices = np.random.choice(
        X_train.shape[0], int(X_train.shape[0] * 0.2), replace=False
    )

    training_loader = DataLoader(
        DataSet(X_train[training_indices], Y_train[training_indices]),
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    mask = np.ones(X_train.shape[0], dtype=bool)
    mask[training_indices] = False

    validation_loader = DataLoader(
        DataSet(X_train[mask], Y_train[mask]),
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    train_network(model, training_loader, validation_loader)

    return model


def __run_data_diagnostics(data: CSVData, labels: CSVData, header: CSVHeader) -> None:
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
    parser.add_argument("--diagnose", action="store_true", help="enable data diagnostics")
    parser.add_argument(
        "--pca",
        action="store_true",
        help="whether to use PCA",
    )
    parser.add_argument(
        "--model",
        choices=["xgb", "nn"],
        default="xgb",
        help="the choice of model to train",
    )
    subparsers = parser.add_subparsers(dest="mode", help="the mode of operation")

    # Sub-parser for k-fold cross-validation
    eval_parser = subparsers.add_parser(
        "eval", description="evaluate using k-fold cross-validation"
    )
    eval_parser.add_argument(
        "-k",
        "--cross-val",
        type=int,
        default=10,
        help="the k for k-fold cross-validation",
    )

    # Sub-parser for final training
    final_parser = subparsers.add_parser(
        "final", description="do final training to generate output"
    )
    final_parser.add_argument(
        "--output",
        type=str,
        default="dist/submission1.csv",
        help="the path by which to save the output CSV (only used in the 'final' mode)",
    )

    __main(parser.parse_args())
