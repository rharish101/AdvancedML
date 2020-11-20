#!/usr/bin/env python
"""The entry point for the scripts for Task 3."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from biosppy.signals.ecg import ecg
from hyperopt import STATUS_FAIL, STATUS_OK, fmin, hp, tpe
from imblearn.over_sampling import ADASYN
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
from typing_extensions import Final
from xgboost import XGBClassifier

from typings import BaseClassifier, CSVData
from utilities.data import read_csv, run_data_diagnostics
from utilities.model import evaluate_model, finalize_model, visualize_model

TASK_DATA_DIRECTORY: Final[str] = "data/task3"
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
    "focus": hp.lognormal("focus", 1.0, 1.0),
    "alpha_1": hp.lognormal("alpha_1", 1.0, 1.0),
    "alpha_2": hp.lognormal("alpha_2", 1.0, 1.0),
}
SVM_SPACE: Final = {"C": hp.lognormal("C", 1.0, 1.0)}
ENSEMBLE_SPACE: Final = {
    "svm_wt": hp.lognormal("svm_wt", 1.0, 1.0),
    **XGB_SPACE,
    **SVM_SPACE,
}
MODEL_SPACE: Final = {"xgb": XGB_SPACE, "svm": SVM_SPACE, "ensemble": ENSEMBLE_SPACE}
SMOTE_SPACE: Final = {
    "sampling_0": hp.uniform("sampling_0", 0, 1),
    "sampling_2": hp.uniform("sampling_2", 0, 1),
    "k_neighbors": hp.choice("k_neighbors", range(1, 10)),
}
LOC_SPACE: Final = {
    "n_neighbors": hp.choice("n_neighbors", range(1, 30)),
    "contamination": hp.quniform("contamination", 0.05, 0.5, 0.05),
}
ISOLATION_SPACE: Final = {
    "n_estimators": hp.choice("n_estimators", range(30, 150)),
    "contamination": hp.quniform("contamination", 0.05, 0.5, 0.05),
}
OUTLIER_SPACE: Final = {"loc": LOC_SPACE, "isolation": ISOLATION_SPACE}

SAMPLING_RATE: Final = 300.0


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

    X_train = get_ecg_features(X_train)

    if args.mode == "tune":
        print("Starting hyper-parameter tuning")
        smote_space = SMOTE_SPACE if args.smote else {}
        outlier_space = OUTLIER_SPACE[args.outlier] if args.outlier is not None else {}
        space = {**MODEL_SPACE[args.model], **smote_space, **outlier_space}

        saved_config = {}

        if args.extend:
            with open(args.config, "r") as conf_file:
                saved_config = yaml.safe_load(conf_file)
            saved_config = {} if saved_config is None else saved_config

            space = {
                k: v
                for k, v in space.items()
                if not any(str(k2) in str(k) for k2, _ in saved_config.items())
            }

        best = fmin(
            lambda config: objective(
                X_train,
                Y_train,
                args.model,
                args.smote,
                args.outlier,
                {**config, **saved_config},
            ),
            space,
            algo=tpe.suggest,
            max_evals=args.max_evals,
            rstate=np.random.RandomState(0),
        )

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

    smote_fn = get_smote_fn(**config) if args.smote else None
    model = choose_model(args.model, **config)
    outlier_detection = (
        get_outlier_detection(**config) if args.outlier is not None else None  # type:ignore
    )

    if args.mode == "eval":
        score = evaluate_model(
            model,
            X_train,
            Y_train,
            k=args.cross_val,
            smote_fn=smote_fn,
            outlier_detection=outlier_detection,
        )

        print(f"Average balanced accuracy is: {score:.4f}")

    elif args.mode == "final":
        X_test, _ = read_csv(f"{args.data_dir}/{TEST_DATA_PATH}")
        if X_test is None:
            raise RuntimeError("There was a problem with reading CSV data")

        # Save test IDs as we need to add them to the submission file
        test_ids = X_test[:, 0]
        X_test = X_test[:, 1:]

        X_test = get_ecg_features(X_test)

        finalize_model(
            model,
            X_train,
            Y_train,
            X_test,
            test_ids,
            args.output,
            smote_fn=smote_fn,
            outlier_detection=outlier_detection,
        )

        if args.visual:
            visualize_model(model, X_train, Y_train)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


def get_ecg_features(X_train: CSVData) -> np.ndarray:
    """Get average ECG heart beats."""
    X_train_features = []

    for x in X_train:
        x = np.array([v for v in x if not np.isnan(v)])
        beats = ecg(x, sampling_rate=SAMPLING_RATE, show=False)["templates"]
        median = np.median(beats, axis=0)
        mad = median_abs_deviation(beats, axis=0)
        X_train_features.append(np.append(median, mad).tolist())

    return np.array(X_train_features)


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


def objective(
    X_train: CSVData,
    Y_train: CSVData,
    model: str,
    smote: bool,
    outlier: bool,
    config: Dict[str, Union[float, int]],
) -> Dict[str, Any]:
    """Get the objective function for Hyperopt."""
    try:
        smote_fn = get_smote_fn(**config) if smote else None  # type:ignore
        model = choose_model(model, **config)  # type:ignore

        outlier_detection = (
            get_outlier_detection(**config) if outlier is not None else None  # type:ignore
        )

        # Keep k low for faster evaluation
        score = evaluate_model(
            model,
            X_train,
            Y_train,
            k=5,
            smote_fn=smote_fn,
            outlier_detection=outlier_detection,
        )

        # We need to maximize score, so minimize the negative
        return {"loss": -score, "status": STATUS_OK}

    except Exception:
        return {"loss": 0, "status": STATUS_FAIL}


def get_outlier_detection(
    contamination: float,
    n_estimators: Optional[int] = None,
    n_neighbors: Optional[int] = None,
    **kwargs,
) -> Any:
    """Get outlier detection model given the hyper-parameters."""
    if n_neighbors is not None:
        return LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
    elif n_estimators is not None:
        return IsolationForest(
            contamination=contamination, n_estimators=n_estimators, random_state=0
        )

    return None


def get_smote_fn(
    sampling_0: Optional[float] = None,
    sampling_2: Optional[float] = None,
    k_neighbors: int = 5,
    **kwargs,
) -> ADASYN:
    """Get a function that chooses the SMOTE model given the hyper-parameters."""
    # Hardcoding class 1 to be majority
    if sampling_0 is None or sampling_2 is None:
        sampling_strategy: Optional[List[float]] = None
    else:
        sampling_strategy = [sampling_0, 1.0, sampling_2]

    def smote_fn(Y):
        counts = np.bincount(Y.astype(np.int64))
        if sampling_strategy is None:
            sampling_strategy_full = "auto"
        else:
            sampling_strategy_full = {
                i: int(sampling_strategy[i] * (max(counts) - counts[i]) + counts[i])
                for i in range(3)
            }
        return ADASYN(
            sampling_strategy=sampling_strategy_full, n_neighbors=k_neighbors, random_state=0
        )

    return smote_fn


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
        random_state=0,
    )

    xgb_model = XGBClassifier(random_state=0)

    svm_model = SVC(C=C, class_weight="balanced", random_state=0)

    random_forest_classifier = RandomForestClassifier()

    if name == "xgb":
        return xgb_model
    elif name == "svm":
        return svm_model
    elif name == "ensemble":
        model_wt = np.array([1.0, svm_wt])
        model_wt /= sum(model_wt)
        return VotingClassifier([("xgb", xgb_model), ("svm", svm_model)], weights=model_wt)
    elif name == "forest":
        return random_forest_classifier
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
        default="data/task3",
        help="path to the directory containing the task data",
    )
    parser.add_argument("--diagnose", action="store_true", help="enable data diagnostics")
    parser.add_argument("--smote", action="store_true", help="use SMOTE")
    parser.add_argument(
        "--outlier",
        choices=["loc", "isolation"],
        help="the choice of model to use for outlier detection",
    )
    parser.add_argument(
        "--model",
        choices=["xgb", "svm", "ensemble", "forest"],
        default="xgb",
        help="the choice of model to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/task3.yaml",
        help="the path to the YAML config for the hyper-parameters",
    )
    subparsers = parser.add_subparsers(dest="mode", help="the mode of operation")

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
        default="dist/submission3.csv",
        help="the path by which to save the output CSV",
    )
    parser.add_argument("--visual", action="store_true", help="enable model visualizations")

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
    tune_parser.add_argument(
        "--extend", action="store_true", help="only tune parameters not present in config file"
    )

    __main(parser.parse_args())
