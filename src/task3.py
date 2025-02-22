#!/usr/bin/env python
"""The entry point for the scripts for Task 3."""
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml
from biosppy.signals.ecg import ecg
from hyperopt import STATUS_FAIL, STATUS_OK, fmin, hp, tpe
from imblearn.over_sampling import RandomOverSampler
from scipy.fft import fft
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
from tqdm import tqdm
from tsfresh.feature_extraction.feature_calculators import (
    absolute_sum_of_changes,
    change_quantiles,
    fft_aggregated,
    mean_abs_change,
    variance,
    variation_coefficient,
)
from typing_extensions import Final
from xgboost import XGBClassifier

from models.task3_nn import NN
from typings import BaseClassifier, CSVData
from utilities.data import read_csv
from utilities.model import evaluate_model, feature_selection, finalize_model, visualize_model

TRAINING_DATA_CSV: Final = "X_train.csv"
TRAINING_LABELS_CSV: Final = "y_train.csv"
TEST_DATA_CSV: Final = "X_test.csv"

TRAINING_FEAT_NPY: Final = "train-features.npy"
TEST_FEAT_NPY: Final = "test-features.npy"

# Search distributions for hyper-parameters
XGB_SPACE: Final = {
    "n_estimators": hp.choice("n_estimators", range(10, 500)),
    "max_depth": hp.choice("max_depth", range(2, 20)),
    "xgb_lr": hp.quniform("xgb_lr", 0.01, 1.0, 0.01),
    "gamma_xgb": hp.uniform("gamma_xgb", 0.0, 1.0),
    "min_child_weight": hp.uniform("min_child_weight", 1, 8),
    "subsample": hp.uniform("subsample", 0.8, 1),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1.0, 0.05),
    "reg_lambda": hp.lognormal("reg_lambda", 1.0, 1.0),
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
    "n_neighbors": hp.choice("n_neighbors", range(30, 150)),
    "contamination": hp.quniform("contamination", 0.05, 0.2, 0.05),
}
ISOLATION_SPACE: Final = {
    "n_estimators": hp.choice("n_estimators", range(30, 150)),
    "contamination": hp.quniform("contamination", 0.05, 0.5, 0.05),
}
OUTLIER_SPACE: Final = {"loc": LOC_SPACE, "isolation": ISOLATION_SPACE}

SAMPLING_RATE: Final = 300.0


def __main(args: Namespace) -> None:
    if not os.path.exists(args.features_dir):
        os.makedirs(args.features_dir)

    X_train = get_ecg_features(
        os.path.join(args.data_dir, TRAINING_DATA_CSV),
        os.path.join(args.features_dir, TRAINING_FEAT_NPY),
    )

    # Read in data
    Y_train, _ = read_csv(os.path.join(args.data_dir, TRAINING_LABELS_CSV))
    if Y_train is None:
        raise RuntimeError("There was a problem with reading CSV data")
    # Remove training IDs, as they are in sorted order for training data
    Y_train = Y_train[:, 1]

    # Load hyper-parameters, if a config exists
    with open(args.config, "r") as conf_file:
        config = yaml.safe_load(conf_file)
    config = {} if config is None else config

    model = choose_model(args.model, args.log_dir, **config)

    if args.select_features:
        selected = feature_selection(model, X_train, Y_train, args.selected_features_path, k=5)
        X_train = X_train[:, selected]

    if args.mode == "tune":
        print("Starting hyper-parameter tuning")
        smote_space = SMOTE_SPACE if args.smote else {}
        outlier_space = OUTLIER_SPACE[args.outlier] if args.outlier is not None else {}
        space = {**MODEL_SPACE[args.model], **smote_space, **outlier_space}

        if args.extend:
            space = {key: val for key, val in space.items() if key not in config}
        else:
            config = {}

        best = fmin(
            lambda hp_config: objective(
                X_train,
                Y_train,
                args.model,
                args.log_dir,
                args.smote,
                args.outlier,
                {**hp_config, **config},
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

    smote_fn = get_smote_fn(**config) if args.smote else None
    outlier_detection = (
        get_outlier_detection(args.outlier, **config)
        if args.outlier is not None
        else None  # type:ignore
    )

    if args.mode == "eval":
        train_score, val_score = evaluate_model(
            model,
            X_train,
            Y_train,
            k=args.cross_val,
            smote_fn=smote_fn,
            outlier_detection=outlier_detection,
            single=args.single,
            visualize=args.visual,
        )

        print(f"Micro-average F1 training score is: {train_score:.4f}")
        print(f"Micro-average F1 validation score is: {val_score:.4f}")

    elif args.mode == "final":
        X_test = get_ecg_features(
            os.path.join(args.data_dir, TEST_DATA_CSV),
            os.path.join(args.features_dir, TEST_FEAT_NPY),
        )

        if args.select_features:
            X_test = X_test[:, selected]

        # Assuming test IDs as in ascending order
        test_ids = np.arange(len(X_test))

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


def get_ecg_features(raw_path: str, transformed_path: str) -> np.ndarray:
    """Get ECG features from the raw data or the saved transformed data."""
    if os.path.exists(transformed_path):
        print("Loading features from %s..." % transformed_path)
        return np.load(transformed_path)

    raw_data, _ = read_csv(raw_path)
    if raw_data is None:
        raise RuntimeError("There was a problem with reading CSV data")
    # Remove the IDs
    raw_data = raw_data[:, 1:]

    ecg_features = []
    print("Transforming signal with biosppy...")

    for x in tqdm(raw_data):
        x = np.array([i for i in x if not np.isnan(i)])
        extracted = ecg(x, sampling_rate=SAMPLING_RATE, show=False)
        beats = fft(extracted["templates"])

        # We don't use the given heart rate as it applies physiological limits
        # (40 <= heart rate <= 200), and thus this may be empty. We also don't care about
        # smoothing it, as we're using median and MAD anyway.
        heart_rate = SAMPLING_RATE * (60.0 / np.diff(extracted["rpeaks"]))
        # Duplicate last element so that heart_rate and beats are of same length
        heart_rate = np.append(heart_rate, heart_rate[-1]).reshape(-1, 1)
        ecg_features.append(np.hstack((np.real(beats), np.imag(beats), heart_rate)))

    final_features = np.hstack(
        (
            extract_statistics(ecg_features),
            extract_heartrate_tsfresh(ecg_features),
        )
    )

    np.save(transformed_path, final_features)
    return final_features


def extract_heartrate_tsfresh(transformed: np.ndarray) -> np.ndarray:
    """Extract all tsfresh features from heart rate."""
    ecg_features = []
    print("Extracting TSFRESH statistics from heart rate signals...")

    for x in tqdm(transformed):
        vchange_quantiles_abs = change_quantiles(x[:, -1], 0, 0.8, True, "var")
        vfft_aggregated_k = list(fft_aggregated(x[:, -1], [{"aggtype": "kurtosis"}]))[0][1]
        vmean_abs_change = mean_abs_change(x[:, -1])
        vabsolute_sum_of_changes = absolute_sum_of_changes(x[:, -1])
        vfft_aggregated_s = list(fft_aggregated(x[:, -1], [{"aggtype": "skew"}]))[0][1]
        vfft_aggregated_c = list(fft_aggregated(x[:, -1], [{"aggtype": "centroid"}]))[0][1]
        vvariance = variance(x[:, -1])
        vvariation_coefficient = variation_coefficient(x[:, -1])

        new_tsfresh = np.array(
            [
                vchange_quantiles_abs,
                vfft_aggregated_k,
                vmean_abs_change,
                vabsolute_sum_of_changes,
                vfft_aggregated_s,
                vfft_aggregated_c,
                vvariance,
                vvariation_coefficient,
            ]
        )

        ecg_features.append(np.concatenate(new_tsfresh, axis=0))

    return np.array(ecg_features)


def extract_statistics(transformed: np.ndarray) -> np.ndarray:
    """Extract median and deviation statistics from the transformed ECG signals."""
    ecg_features = []
    print("Extracting statistics from transformed signals...")

    for x in tqdm(transformed):
        median_temp = np.median(x[:, :-1], axis=0)
        mad_temp = median_abs_deviation(x[:, :-1], axis=0)

        median_hr = np.median(x[:, -1], keepdims=True)
        mad_hr = median_abs_deviation(x[:, -1]).reshape([-1])

        features = np.concatenate([median_temp, mad_temp, median_hr, mad_hr])
        ecg_features.append(features)

    return np.array(ecg_features)


def objective(
    X_train: CSVData,
    Y_train: CSVData,
    model: str,
    log_dir: str,
    smote: bool,
    outlier: str,
    config: Dict[str, Union[float, int]],
) -> Dict[str, Any]:
    """Get the objective function for Hyperopt."""
    try:
        smote_fn = get_smote_fn(**config) if smote else None  # type:ignore
        model = choose_model(model, log_dir, **config)  # type:ignore

        outlier_detection = (
            get_outlier_detection(outlier, **config) if outlier is not None else None  # type:ignore
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
    name: str,
    contamination: float,
    n_estimators: Optional[int] = 100,
    n_neighbors: Optional[int] = 100,
    **kwargs,
) -> Any:
    """Get outlier detection model given the hyper-parameters."""
    if name == "loc":
        return LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
    elif name == "isolation":
        return IsolationForest(
            contamination=contamination, n_estimators=n_estimators, random_state=0
        )

    return None


def get_smote_fn(
    sampling_0: Optional[float] = None,
    sampling_2: Optional[float] = None,
    **kwargs,
) -> RandomOverSampler:
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
        return RandomOverSampler(sampling_strategy=sampling_strategy_full, random_state=0)

    return smote_fn


def choose_model(
    name: str,
    log_dir: str = "logs",
    n_estimators: int = 100,
    max_depth: int = 6,
    xgb_lr: float = 0.3,
    gamma_xgb: float = 0.0,
    min_child_weight: float = 1.0,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_lambda: float = 1.0,
    C: float = 1.0,
    nn_wt: float = 1.0,
    epochs: int = 50,
    batch_size: int = 64,
    nn_lr: float = 1e-3,
    lr_step: int = 10000,
    lr_decay: float = 0.75,
    weight_decay: float = 1e-3,
    balance_weights: bool = True,
    **kwargs,
) -> BaseClassifier:
    """Choose a model given the name and hyper-parameters."""
    xgb_model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=xgb_lr,
        gamma=gamma_xgb,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=0,
    )
    svm_model = SVC(C=C, class_weight="balanced", random_state=0)
    random_forest_classifier = RandomForestClassifier()

    nn_model = NN(
        epochs=epochs,
        batch_size=batch_size,
        log_dir=log_dir,
        learning_rate=nn_lr,
        lr_step=lr_step,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        balance_weights=balance_weights,
        random_state=0,
    )

    if name == "xgb":
        return xgb_model
    elif name == "svm":
        return svm_model
    elif name == "ensemble":
        model_wt = np.array([1.0, nn_wt])
        model_wt /= sum(model_wt)
        return VotingClassifier(
            [("xgb", xgb_model), ("nn", nn_model)], voting="soft", weights=model_wt
        )
    elif name == "forest":
        return random_forest_classifier
    elif name == "nn":
        return nn_model
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
    parser.add_argument(
        "--features-dir",
        type=str,
        default="config/task3/",
        help="where the extracted features are stored or should be stored",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/task3",
        help="path to the directory where to dump logs",
    )
    parser.add_argument("--smote", action="store_true", help="use SMOTE")
    parser.add_argument(
        "--outlier",
        choices=["loc", "isolation"],
        help="the choice of model to use for outlier detection",
    )
    parser.add_argument(
        "--model",
        choices=["xgb", "svm", "ensemble", "forest", "nn"],
        default="xgb",
        help="the choice of model to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/task3.yaml",
        help="the path to the YAML config for the hyper-parameters",
    )
    parser.add_argument(
        "--keep-features",
        dest="select_features",
        action="store_false",
        help="whether to skip training the most optimal features",
    )
    parser.add_argument(
        "--selected-features-path",
        type=str,
        default="config/features-task3.txt",
        help="where the most optimal features are stored",
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
    eval_parser.add_argument(
        "--single",
        action="store_true",
        help="whether to evaluate only on a single fold (ie standard cross-validation)",
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
