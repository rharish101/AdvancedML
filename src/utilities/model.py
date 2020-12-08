"""Utility functions for model-related tasks."""
import os
from typing import Any, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    balanced_accuracy_score,
    f1_score,
    plot_confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.random import sample_without_replacement

from typings import BaseRegressor, CSVData
from utilities.data import create_submission_file

SamplerFnType = Optional[Callable[[CSVData], BaseSampler]]

def evaluate_model_task4(
    model: BaseEstimator,
    X_train: CSVData,
    Y_train: CSVData,
    k: int,
    smote_fn: SamplerFnType = None,
    outlier_detection: Any = None,
    single: bool = False,
    visualize: bool = False,
) -> Tuple[float, float]:
    """Perform cross-validation on the given dataset and return the R^2 score.

    Parameters
    ----------
    model: The model
    X_train: The training data
    Y_train: The training labels
    k: The number of folds in k-fold cross-validation (i.e. the number of subjects)
    smote_fn: The function that takes labels and returns SMOTE
    single: Whether to evaluate only on a single fold (ie standard cross-validation)

    Returns
    -------
    The training score
    The validation score
    """

    train_score = 0
    val_score = 0

    X_train = np.vstack([np.split(X_train, k)])
    Y_train = np.vstack([np.split(Y_train, k)])

    for test_index in range(k):
        X_train_cv, X_test_cv = np.concatenate((X_train[(test_index + 1) % k], X_train[(test_index + 2) % k])), X_train[test_index]
        Y_train_cv, Y_test_cv = np.concatenate((Y_train[(test_index + 1) % k], Y_train[(test_index + 2) % k])), Y_train[test_index]

        print(X_train_cv.shape)
        print(X_test_cv.shape)

        if outlier_detection is not None:
            outliers = outlier_detection.fit_predict(X_train_cv)
            X_train_cv = X_train_cv[outliers == 1]
            Y_train_cv = Y_train_cv[outliers == 1]

        if smote_fn:
            smote = smote_fn(Y_train_cv)
            X_train_cv, Y_train_cv = smote.fit_resample(X_train_cv, Y_train_cv)

        try:
            model.fit(X_train_cv, Y_train_cv)
        except KeyboardInterrupt:
            pass

        train_pred = model.predict(X_train_cv)
        train_score += f1_score(Y_train_cv, train_pred, average="micro")

        test_pred = model.predict(X_test_cv)
        val_score += f1_score(Y_test_cv, test_pred, average="micro")

        if visualize:
            print(f"\nComputing training statistics for fold {fold_index + 1}/{k} ...")
            create_visualization(model, X_train_cv, Y_train_cv, "Training Metrics")

            print(f"\nComputing validation statistics for fold {fold_index + 1}/{k} ...")
            create_visualization(model, X_test_cv, Y_test_cv, "Validation Metrics")

            plt.show()

        if single:
            return train_score, val_score

    return train_score / k, val_score / k

def evaluate_model(
    model: BaseEstimator,
    X_train: CSVData,
    Y_train: CSVData,
    k: int,
    smote_fn: SamplerFnType = None,
    outlier_detection: Any = None,
    single: bool = False,
    visualize: bool = False,
) -> Tuple[float, float]:
    """Perform cross-validation on the given dataset and return the R^2 score.

    Parameters
    ----------
    model: The model
    X_train: The training data
    Y_train: The training labels
    k: The number of folds in k-fold cross-validation
    smote_fn: The function that takes labels and returns SMOTE
    single: Whether to evaluate only on a single fold (ie standard cross-validation)

    Returns
    -------
    The training score
    The validation score
    """
    train_score = 0
    val_score = 0
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    for fold_index, (train_index, test_index) in enumerate(kf.split(X_train, Y_train)):
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        Y_train_cv, Y_test_cv = Y_train[train_index], Y_train[test_index]

        if outlier_detection is not None:
            outliers = outlier_detection.fit_predict(X_train_cv)
            X_train_cv = X_train_cv[outliers == 1]
            Y_train_cv = Y_train_cv[outliers == 1]

        if smote_fn:
            smote = smote_fn(Y_train_cv)
            X_train_cv, Y_train_cv = smote.fit_resample(X_train_cv, Y_train_cv)

        try:
            model.fit(X_train_cv, Y_train_cv)
        except KeyboardInterrupt:
            pass

        train_pred = model.predict(X_train_cv)
        train_score += f1_score(Y_train_cv, train_pred, average="micro")

        test_pred = model.predict(X_test_cv)
        val_score += f1_score(Y_test_cv, test_pred, average="micro")

        if visualize:
            print(f"\nComputing training statistics for fold {fold_index + 1}/{k} ...")
            create_visualization(model, X_train_cv, Y_train_cv, "Training Metrics")

            print(f"\nComputing validation statistics for fold {fold_index + 1}/{k} ...")
            create_visualization(model, X_test_cv, Y_test_cv, "Validation Metrics")

            plt.show()

        if single:
            return train_score, val_score

    return train_score / k, val_score / k


def evaluate_model_balanced_ensemble(
    model: BaseEstimator,
    X_train: CSVData,
    Y_train: CSVData,
    k: int,
) -> float:
    """Perform cross-validation on the given dataset using ensembling to counteract class imbalance.

    Parameters
    ----------
    model: The model
    X_train: The training data
    Y_train: The training labels
    k: The number of folds in k-fold cross-validation

    Returns
    -------
    The validation score
    """
    w_accuracies = []
    nr_of_classes = len(np.unique(Y_train))
    majority_class = max(set(Y_train), key=list(Y_train).count)

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
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


def finalize_model(
    model: BaseEstimator,
    X_train: CSVData,
    Y_train: CSVData,
    X_test: CSVData,
    test_ids: CSVData,
    output: str,
    smote_fn: SamplerFnType = None,
    outlier_detection: Any = None,
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
    smote_fn: The function that takes labels and returns SMOTE
    """
    print("Training model...")

    if outlier_detection is not None:
        outliers = outlier_detection.fit_predict(X_train)
        X_train = X_train[outliers == 1]
        Y_train = Y_train[outliers == 1]

    if smote_fn:
        smote = smote_fn(Y_train)
        X_train, Y_train = smote.fit_resample(X_train, Y_train)

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


def feature_selection(
    model: BaseRegressor, X_train: CSVData, Y_train: CSVData, features_path: str, k: int
) -> List[bool]:
    """Determine the features yielding best score, and save them.

    Parameters
    ----------
    model: The model that one wishes to use
    X_train: The training data
    Y_train: The training labels
    features_path: The path where to save the features
    k: The number of folds in k-fold cross-validation

    Returns
    -------
    The list of booleans indicating which features to select
    """
    if os.path.exists(features_path):
        print(f"Loading feature selection from {features_path}...")
        return [bool(i) for i in np.loadtxt(features_path, dtype=int)]

    print("Performing feature selection...")
    rec_sel = RFECV(model, step=5, cv=k, verbose=1)
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


def create_visualization(model, X_test, Y_test, title="Model Metrics"):
    """Compute the metrics and creates the figures used for model visualization.

    Parameters
    ----------
    model: The trained model to get the metrics from

    X_test: Data used to evaluate the model

    Y_test: Data labels for the evaluation data

    title: Title of the figure on which visualizations will be drawn
    """
    try:
        Y_probabilities = model.predict_proba(X_test)
    except AttributeError:
        # Must be an SVM w/o probability=True
        Y_probabilities = model.decision_function(X_test)

    figure, axes = plt.subplots(ncols=3, figsize=(16, 5))
    figure.suptitle(title)

    # Plot confusion matrix
    labels = unique_labels(Y_test)
    plot_confusion_matrix(model, X_test, Y_test, ax=axes[0], normalize="true", labels=labels)

    # Plot precision-recall and ROC curve for each class
    for index, class_label in enumerate(labels):
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


def visualize_model(model, X_test, Y_test):
    """Visualize metrics of a model.

    Parameters
    ----------
    model: The trained model to get the metrics from

    X_test: Data used to evaluate the model

    Y_test: Data labels for the evaluation data

    plot: Matplotlib plot used to visualize the data. A new one will be created if None is given.
    """
    create_visualization(model, X_test, Y_test)
    plt.show()
