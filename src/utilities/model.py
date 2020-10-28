"""Utility functions for model-related tasks."""

import os
from datetime import datetime
from typing import Any, List
from warnings import warn

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from tensorboardX import SummaryWriter

from typings import BaseRegressor, CSVData
from utilities.data import create_submission_file

tensorboard_writer = SummaryWriter(
    log_dir="logs/training" + datetime.now().strftime("-%Y%m%d-%H%M%S")
)


def choose_model(
    name: str,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.3,
    gamma: float = 0.0,
    min_child_weight: float = 1.0,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_lambda: float = 1.0,
) -> BaseRegressor:
    """Choose a model given the name and hyper-parameters."""
    if name == "xgb":
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            gamma=gamma,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
        )
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


def finalize_model(
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

    def log_to_tensorboard(env):
        for name, value in env.evaluation_result_list:
            tensorboard_writer.add_scalar(name, value, env.iteration)

    model.fit(X_train, Y_train, eval_set=[(X_train, Y_train)], callbacks=[log_to_tensorboard])
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
