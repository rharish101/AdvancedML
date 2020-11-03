#!/usr/bin/env python
"""The entry point for the scripts for Task 2."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Dict, Tuple, Union

import numpy as np
import yaml
from hyperopt import fmin, hp, tpe
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from typing_extensions import Final
from xgboost import XGBClassifier

from typings import BaseClassifier
from utilities.data import read_csv, run_data_diagnostics
from utilities.model import (
    evaluate_model,
    evaluate_model_balanced_ensemble,
    finalize_model,
    finalize_model_balanced_ensemble,
    visualize_model,
)

TASK_DATA_DIRECTORY: Final[str] = "data/task2"
TRAINING_DATA_NAME: Final[str] = "X_train.csv"
TRAINING_LABELS_NAME: Final[str] = "y_train.csv"
TEST_DATA_PATH: Final[str] = "X_test.csv"

# Search distributions for hyper-parameters
XGB_SPACE: Final = {
    "n_estimators": hp.choice("n_estimators", range(10, 500)),
    "max_depth": hp.choice("max_depth", range(2, 20)),
    "learning_rate": hp.quniform("learning_rate", 0.01, 1.0, 0.01),
    "gamma_xgb": hp.uniform("gamma_xgb", 0.0, 1.0),
    "min_child_weight": hp.uniform("min_child_weight", 1, 8),
    "subsample": hp.uniform("subsample", 0.8, 1),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1.0, 0.05),
    "reg_lambda": hp.lognormal("reg_lambda", 1.0, 1.0),
}
SVM_SPACE: Final = {
    "C": hp.lognormal("C", 1.0, 1.0),
    "kernel": hp.choice("kernel", ["poly", "rbf", "sigmoid"]),
    "gamma_svm": hp.uniform("gamma_svm", 1e-4, 1.0),
}
ENSEMBLE_SPACE: Final = {
    "svm_wt": hp.lognormal("svm_wt", 1.0, 1.0),
    **XGB_SPACE,
    **SVM_SPACE,
}
SPACE: Final = {
    "focus": hp.lognormal("focus", 1.0, 1.0),
    "alpha_1": hp.lognormal("alpha_1", 1.0, 1.0),
    "alpha_2": hp.lognormal("alpha_2", 1.0, 1.0),
    "model": hp.choice(
        "model", [("xgb", XGB_SPACE), ("svm", SVM_SPACE), ("ensemble", ENSEMBLE_SPACE)]
    ),
}


def __main(args: Namespace) -> None:
    # Read in data
    X_train, X_header = read_csv(f"{args.data_dir}/{TRAINING_DATA_NAME}")
    Y_train, _ = read_csv(f"{args.data_dir}/{TRAINING_LABELS_NAME}")

    if X_train is None or Y_train is None:
        raise RuntimeError("There was a problem with reading CSV data")

    if args.diagnose:
        run_data_diagnostics(X_train, Y_train, header=X_header or ())

    # Remove training IDs, as they are in sorted order for training data
    X_train = X_train[:, 1:]
    Y_train = Y_train[:, 1]

    if args.mode == "tune":

        def objective(config: Dict[str, Union[float, int]]) -> float:
            model = choose_model(config["model"][0], **config["model"][1], **config)  # type:ignore
            # Keep k low for faster evaluation
            if args.balanced_ensemble:
                score = evaluate_model_balanced_ensemble(
                    model, X_train, Y_train, k=5, scoring="balanced_accuracy"
                )
            else:
                score = evaluate_model(
                    model, X_train, Y_train, k=5, scoring="balanced_accuracy", smote=args.smote
                )

            # We need to maximize score, so minimize the negative
            return -score

        print("Starting hyper-parameter tuning")
        best = fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.max_evals)

        # Convert numpy dtypes to native Python
        for key, value in best.items():
            best[key] = value.item()

        with open(args.config, "w") as conf_file:
            yaml.dump(best, conf_file)
        print(f"Best parameters saved in {args.config}")
        return

    # Load hyper-parameters, if a config exists
    with open(args.config, "r") as conf_file:
        config = yaml.safe_load(conf_file)
    config = {} if config is None else config

    model = choose_model(args.model, **config)

    if args.mode == "eval":
        if args.balanced_ensemble:
            score = evaluate_model_balanced_ensemble(
                model, X_train, Y_train, k=args.cross_val, scoring="balanced_accuracy"
            )
        else:
            score = evaluate_model(
                model,
                X_train,
                Y_train,
                k=args.cross_val,
                scoring="balanced_accuracy",
                smote=args.smote,
            )

        print(f"Average balanced accuracy is: {score:.4f}")

    elif args.mode == "final":
        X_test, _ = read_csv(f"{args.data_dir}/{TEST_DATA_PATH}")
        if X_test is None:
            raise RuntimeError("There was a problem with reading CSV data")

        # Save test IDs as we need to add them to the submission file
        test_ids = X_test[:, 0]
        X_test = X_test[:, 1:]

        if args.balanced_ensemble:
            finalize_model_balanced_ensemble(model, X_train, Y_train, X_test, test_ids, args.output)
        else:
            finalize_model(model, X_train, Y_train, X_test, test_ids, args.output, smote=args.smote)

        if args.visual:
            visualize_model(model, X_train, Y_train)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


def __loss(
    y_true: np.ndarray, y_pred: np.ndarray, focus: float, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Focal loss: https://arxiv.org/abs/1708.02002."""
    one_hot = np.zeros_like(y_pred)
    one_hot[np.arange(len(y_true)), y_true.astype(np.int)] = 1

    soft = np.exp(y_pred - y_pred.max(1, keepdims=True))
    soft /= soft.sum(1, keepdims=True)
    soft = np.maximum(soft, np.finfo(soft.dtype).eps)  # prevent log(0)

    diff = one_hot - soft
    one_m_soft = np.maximum(1 - soft, np.finfo(soft.dtype).eps)  # prevent div-by-0
    weights = weights.reshape(1, -1)  # 2D for broadcasting

    grad = focus * one_m_soft ** (focus - 1) * np.log(soft) * diff
    grad -= one_m_soft ** focus * diff
    grad *= weights

    hess = -focus * (focus - 1) * one_m_soft ** (focus - 2) * np.log(soft) * soft ** 2 * diff ** 2
    hess += 2 * focus * one_m_soft ** (focus - 1) * soft * diff ** 2
    hess += focus * one_m_soft ** (focus - 1) * np.log(soft) * soft * diff * (diff - soft)
    hess += one_m_soft ** focus * soft * diff
    hess = np.maximum(hess, np.finfo(hess.dtype).eps)  # numerical stability
    hess *= weights

    return grad.reshape(-1), hess.reshape(-1)


def choose_model(
    name: str,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.3,
    gamma_xgb: float = 0.0,
    min_child_weight: float = 1.0,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_lambda: float = 1.0,
    focus: float = 1.0,
    alpha_1: float = 1.0,
    alpha_2: float = 1.0,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma_svm: float = 1e-3,
    svm_wt: float = 1.0,
    **kwargs,
) -> BaseClassifier:
    """Choose a model given the name and hyper-parameters."""
    weights = np.array([1.0, alpha_1, alpha_2])  # 2D for broadcasting
    weights /= sum(weights)

    xgb_model = XGBClassifier(
        objective=lambda y_true, y_pred: __loss(y_true, y_pred, focus=focus, weights=weights),
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        gamma=gamma_xgb,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
    )

    svm_model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma_svm,
        class_weight={i: val for i, val in enumerate(weights)},
    )

    if name == "xgb":
        return xgb_model
    elif name == "svm":
        return svm_model
    elif name == "ensemble":
        model_wt = np.array([1.0, svm_wt])
        model_wt /= sum(model_wt)
        return VotingClassifier([("xgb", xgb_model), ("svm", svm_model)], weights=model_wt)
    else:
        raise ValueError(f"Invalid model name: {name}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="The entry point for the scripts for Task 2",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/task2",
        help="path to the directory containing the task data",
    )
    parser.add_argument("--diagnose", action="store_true", help="enable data diagnostics")
    parser.add_argument("--smote", action="store_true", help="use SMOTE")
    parser.add_argument(
        "--balanced-ensemble", action="store_true", help="use ensembling to obtain balanced sets"
    )
    parser.add_argument(
        "--model",
        choices=["xgb", "svm", "ensemble"],
        default="svm",
        help="the choice of model to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/task2.yaml",
        help="the path to the YAML config for the hyper-parameters",
    )
    subparsers = parser.add_subparsers(dest="mode", help="the mode of operation")
    parser.add_argument("--visual", action="store_true", help="enable model visualizations")

    # Sub-parser for k-fold cross-validation
    eval_parser = subparsers.add_parser(
        "eval",
        description="evaluate using k-fold cross-validation",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument(
        "-k",
        "--cross-val",
        type=int,
        default=5,
        help="the k for k-fold cross-validation",
    )

    # Sub-parser for final training
    final_parser = subparsers.add_parser(
        "final",
        description="do final training to generate output",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    final_parser.add_argument(
        "--output",
        type=str,
        default="dist/submission2.csv",
        help="the path by which to save the output CSV",
    )

    # Sub-parser for hyper-param tuning
    tune_parser = subparsers.add_parser(
        "tune", description="hyper-parameter tuning", formatter_class=ArgumentDefaultsHelpFormatter
    )
    tune_parser.add_argument(
        "--max-evals",
        type=int,
        default=100,
        help="max. evaluations for hyper-opt",
    )

    __main(parser.parse_args())
