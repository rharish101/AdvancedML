"""Utility functions for model-related tasks."""
import os
from datetime import datetime
from typing import Any, Callable, List, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    balanced_accuracy_score,
    plot_confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.random import sample_without_replacement
from tensorboardX import SummaryWriter

from typings import BaseRegressor, CSVData
from utilities.data import create_submission_file

tensorboard_writer = SummaryWriter(
    log_dir="logs/training" + datetime.now().strftime("-%Y%m%d-%H%M%S")
)


def evaluate_model_balanced_ensemble(
    model: BaseEstimator,
    X_train: CSVData,
    Y_train: CSVData,
    k: int,
    scoring: Union[str, Callable[[BaseEstimator, CSVData, CSVData], float]],
) -> float:
    """Perform cross-validation on the given dataset using ensembling to counteract class imbalance.

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
    w_accuracies = []
    nr_of_classes = len(np.unique(Y_train))
    majority_class = max(set(Y_train), key=list(Y_train).count)

    kf = StratifiedKFold(n_splits=k, shuffle=True)
    for train_index, test_index in kf.split(X_train, Y_train):
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        Y_train_cv, Y_test_cv = Y_train[train_index], Y_train[test_index]

        X_majority_class = np.array(
            [X_train_cv[i] for i in range(len(Y_train_cv)) if Y_train_cv[i] == majority_class]
        )
        Y_majority_class = np.array(
            [Y_train_cv[i] for i in range(len(Y_train_cv)) if Y_train_cv[i] == majority_class]
        )

        X_minority_class = [
            X_train_cv[i] for i in range(len(Y_train_cv)) if Y_train_cv[i] != majority_class
        ]
        Y_minority_class = [
            Y_train_cv[i] for i in range(len(Y_train_cv)) if Y_train_cv[i] != majority_class
        ]

        preds = []
        for k in range(int(len(X_majority_class) / len(X_minority_class) * (nr_of_classes - 1))):
            indices = sample_without_replacement(
                n_population=len(X_majority_class), n_samples=len(X_minority_class)
            )

            X_train_final = [*X_majority_class[indices], *X_minority_class]

            Y_train_final = [*Y_majority_class[indices], *Y_minority_class]

            model.fit(X_train_final, Y_train_final)
            preds.append(model.predict(X_test_cv))

        # Take majority vote
        majority_votes = [max(set(votes), key=votes.count) for votes in zip(*preds)]
        w_accuracies.append(balanced_accuracy_score(Y_test_cv, majority_votes))

    # Return average balanced accuracy
    return sum(w_accuracies) / len(w_accuracies)


def evaluate_model(
    model: BaseEstimator,
    X_train: CSVData,
    Y_train: CSVData,
    k: int,
    scoring: Union[str, Callable[[BaseEstimator, CSVData, CSVData], float]],
    smote: bool,
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
    score = 0
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    for train_index, test_index in kf.split(X_train, Y_train):
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        Y_train_cv, Y_test_cv = Y_train[train_index], Y_train[test_index]

        if smote:
            sm = SMOTEENN(random_state=17)
            X_train_cv, Y_train_cv = sm.fit_resample(X_train_cv, Y_train_cv)

        model.fit(X_train_cv, Y_train_cv)
        pred = model.predict(X_test_cv)
        score += balanced_accuracy_score(Y_test_cv, pred)

    return score / k

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
    smote: bool,
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

    if smote:
        sm = SMOTEENN(random_state=17)
        X_train, Y_train = sm.fit_resample(X_train, Y_train)

    model.fit(X_train, Y_train)

    print("Model trained")
    Y_pred = model.predict(X_test)
    submission: Any = np.stack([test_ids, Y_pred], 1)  # Add IDs
    create_submission_file(output, submission, header=("id", "y"))


def finalize_model_balanced_ensemble(
    model: BaseEstimator,
    X_train: CSVData,
    Y_train: CSVData,
    X_test: CSVData,
    test_ids: CSVData,
    output: str,
) -> None:
    """Train the model using ensembling to counteract class imbalance.

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

    nr_of_classes = len(np.unique(Y_train))

    majority_class = max(set(Y_train), key=Y_train.count)

    X_majority_class = np.array(
        [X_train[i] for i in range(len(Y_train)) if Y_train[i] == majority_class]
    )
    Y_majority_class = np.array(
        [Y_train[i] for i in range(len(Y_train)) if Y_train[i] == majority_class]
    )

    X_minority_class = [X_train[i] for i in range(len(Y_train)) if Y_train[i] != majority_class]
    Y_minority_class = [Y_train[i] for i in range(len(Y_train)) if Y_train[i] != majority_class]

    preds = []
    for k in range(int(len(X_majority_class) / len(X_minority_class) * (nr_of_classes - 1))):
        indices = sample_without_replacement(
            n_population=len(X_majority_class), n_samples=len(X_minority_class), random_state=17
        )

        X_train_final = [*X_majority_class[indices], *X_minority_class]

        Y_train_final = [*Y_majority_class[indices], *Y_minority_class]

        model.fit(X_train_final, Y_train_final)
        preds.append(model.predict(X_test))

    # Take majority vote
    majority_votes = [max(set(votes), key=votes.count) for votes in zip(*preds)]

    print("Model trained")

    submission: Any = np.stack([test_ids, majority_votes], 1)  # Add IDs
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


def visualize_model(model, X_test, Y_test):
    """Visualize metrics of a model.

    Parameters
    ----------
    model: The trained model to get the metrics from

    X_test: Data used to evaluate the model

    Y_test: Data labels for the evaluation data
    """
    Y_probabilities = model.decision_function(X_test)
    _, axes = plt.subplots(ncols=3, figsize=(16, 5))

    # Plot confusion matrix
    plot_confusion_matrix(model, X_test, Y_test, ax=axes[0])

    # Plot precision-recall and ROC curve for each class
    for index, class_label in enumerate(model.classes_):
        # Plot precision-recall curve
        precision, recall, _ = precision_recall_curve(
            Y_test, Y_probabilities[:, index], pos_label=class_label
        )

        name = f"class {int(class_label)}"
        viz = PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name=name)
        viz.plot(ax=axes[1], name=name)

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(Y_test, Y_probabilities[:, index], pos_label=class_label)
        roc_auc = auc(fpr, tpr)

        viz = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
        viz.plot(ax=axes[2], name=name)

    precisions, recalls, fscores, supports = precision_recall_fscore_support(
        Y_test, model.predict(X_test)
    )

    for index, (precision, recall, fscore, support) in enumerate(
        zip(precisions, recalls, fscores, supports)
    ):
        print(
            f"class {index} - precision: {precision:0.4f}, recall: {recall:0.4f}",
            f"fscore: {fscore:0.4f}, support: {support}",
        )

    plt.show()
