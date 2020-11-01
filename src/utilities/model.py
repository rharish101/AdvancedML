"""Utility functions for model-related tasks."""
import os
from datetime import datetime
from typing import Any, Callable, List, Union
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from tensorboardX import SummaryWriter

from typings import BaseRegressor, CSVData
from utilities.data import create_submission_file

tensorboard_writer = SummaryWriter(
    log_dir="logs/training" + datetime.now().strftime("-%Y%m%d-%H%M%S")
)


def evaluate_model(
    model: BaseEstimator,
    X_train: CSVData,
    Y_train: CSVData,
    k: int,
    scoring: Union[str, Callable[[BaseEstimator, CSVData, CSVData], float]],
) -> float:
    """Perform cross-validation on the given dataset and return the R^2 score.

    Parameters
    ----------
    model: The model
    X_train: The training data
    Y_train: The training labels
    k: The number of folds in k-fold cross-validation
    scoring: The scoring metric to use

    Returns
    -------
    The validation score
    """
    # Returns an array of the k cross-validation R^2 scores
    scores = cross_val_score(model, X_train, Y_train, cv=k, scoring=scoring)
    avg_score = np.mean(scores)
    return avg_score


def finalize_model(
    model: BaseEstimator,
    X_train: CSVData,
    Y_train: CSVData,
    X_test: CSVData,
    test_ids: CSVData,
    output: str,
) -> None:
    """Train the model on the complete data and generate the submission file.

    Parameters
    ----------
    model: The model
    X_train: The training data
    Y_train: The training labels
    X_test: The test data
    test_ids: The IDs for the test data
    output: The path where to dump the output
    """
    print("Training model...")
    model.fit(X_train, Y_train)
    print("Model trained")
    Y_pred = model.predict(X_test)
    submission: Any = np.stack([test_ids, Y_pred], 1)  # Add IDs
    create_submission_file(output, submission, header=("id", "y"))


def read_selected_features(features_path: str, number_of_features: int) -> List[bool]:
    """Read from SELECTED_FEATURES_PATH which features to be select. If nonexistent, selects all.

    Parameters
    ----------
    features_path: The path to the saved features
    number_of_features: The dimensionality of the data

    Returns
    -------
    The list of booleans indicating which features to preserve
    """
    if os.path.exists(features_path):
        return [True if i == 1 else False for i in np.loadtxt(features_path, dtype=int)]
    else:
        warn(f"No saved features found at: {features_path}. All features will be kept.")
        return [True for i in range(number_of_features)]


def feature_selection(
    model: BaseRegressor, X_train: CSVData, Y_train: CSVData, k: int, features_path: str
) -> List[bool]:
    """Determine the features yielding best score, and save them.

    Parameters
    ----------
    model: The model that one wishes to use
    X_train: The training data
    Y_train: The training labels
    k: The number of folds in k-fold cross-validation
    features_path: The path where to save the features

    Returns
    -------
    The list of booleans indicating which features to select
    """
    rec_sel = RFECV(model, step=5, cv=k)
    rec_sel.fit(X_train, Y_train)
    np.savetxt(features_path, rec_sel.support_, fmt="%d")
    return rec_sel.support_


def select_features_correlation(
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
