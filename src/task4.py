#!/usr/bin/env python
"""The entry point for the scripts for Task 4."""
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from hyperopt import STATUS_FAIL, STATUS_OK, fmin, hp, tpe
from imblearn.over_sampling import RandomOverSampler
from scipy.fft import fft
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from typing_extensions import Final
from xgboost import XGBClassifier

from models.task4_nn import NN
from typings import BaseClassifier, BaseTransformer, CSVData
from utilities.data import augment, read_csv
from utilities.model import evaluate_model_task4, feature_selection, finalize_model, visualize_model

# Suffixes will be prefixed with either "train" or "test"
SUFFIX_EEG1_DATA_CSV: Final = "_eeg1.csv"
SUFFIX_EEG2_DATA_CSV: Final = "_eeg2.csv"
SUFFIX_EMG_DATA_CSV: Final = "_emg.csv"
SUFFIX_FEAT_NPY: Final = "-features.npy"
SUFFIX_LBL_NPY: Final = "-labels.npy"

TRAINING_LABELS_CSV: Final = "train_labels.csv"

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


class TemporalScaler(BaseTransformer):
    """Scaler for robust scaling of 3D data with a temporal axis."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the robust scaler."""
        self._scaler = RobustScaler(*args, **kwargs)

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        """Reshape 3D inputs to 2D.

        Parameters
        ----------
        X: The 3D array in NxCxL form

        Returns
        -------
        The 2D array in (N*L)xC form
        """
        X = np.swapaxes(X, 1, 2)  # NxCxL => NxLxC
        return X.reshape(-1, X.shape[2])  # NxLxC => (N*L)xC

    @staticmethod
    def _unflatten(X: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
        """Reshape 2D inputs to 3D.

        Parameters
        ----------
        X: The 2D array in (N*L)xC form
        orig_shape: The original NxCxL shape

        Returns
        -------
        The 3D array in NxCxL form
        """
        X = X.reshape(-1, orig_shape[2], X.shape[1])  # (N*L)xC => NxLxC
        return np.swapaxes(X, 1, 2)  # NxLxC => NxCxL

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> BaseTransformer:
        """Wrap fit for robust scaling."""
        self._scaler = self._scaler.fit(self._flatten(X), y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Wrap transform for robust scaling."""
        return self._unflatten(self._scaler.transform(self._flatten(X)), X.shape)


def __main(args: Namespace) -> None:
    if not os.path.exists(args.features_dir):
        os.makedirs(args.features_dir)

    X_train, Y_train = get_data(args.data_dir, args.features_dir, "train")
    assert Y_train is not None
    if args.augment is not None:
        X_train, Y_train = augment(X_train, Y_train, 3, factor=args.augment)
    X_train = normalize(X_train, 3)
    if args.model != "nn":
        X_train = X_train.reshape(X_train.shape[0], -1)

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
        train_score, val_score = evaluate_model_task4(
            model,
            X_train,
            Y_train,
            k=3,
            smote_fn=smote_fn,
            outlier_detection=outlier_detection,
            single=args.single,
            visualize=args.visual,
        )

        print(f"Micro-average F1 training score is: {train_score:.4f}")
        print(f"Micro-average F1 validation score is: {val_score:.4f}")

    elif args.mode == "final":
        X_test, _ = get_data(args.data_dir, args.features_dir, "test")
        X_test = normalize(X_test, 2)
        if args.model != "nn":
            X_test = X_test.reshape(X_test.shape[0], -1)
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
            header=("Id", "y"),
            label_indexing=1,
            export_int=True,  # save space, as default is too big to upload
        )

        if args.visual:
            visualize_model(model, X_train, Y_train)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


def normalize(X: CSVData, subjects: int) -> CSVData:
    """Normalize the data per-subject.

    This assumes that each subject's data is arranged contiguously.
    """
    if len(X) % subjects != 0:
        raise ValueError("Number of data points not divisble by number of subjects")

    per_sub_size = len(X) // subjects
    parts = []

    for i in range(0, len(X), per_sub_size):
        scaler = TemporalScaler()
        parts.append(scaler.fit_transform(X[i : i + per_sub_size]))

    return np.concatenate(parts, 0)


def get_data(data_dir: str, features_dir: str, mode: str) -> Tuple[CSVData, Optional[CSVData]]:
    """Get the time-series data representing waves preprocessed using Fast Fourier Transform."""
    if mode not in {"train", "test"}:
        raise ValueError(f"Invalid mode: {mode}")

    features_path = os.path.join(features_dir, mode + SUFFIX_FEAT_NPY)
    labels_path = os.path.join(features_dir, mode + SUFFIX_LBL_NPY)
    if os.path.exists(features_path):
        if mode == "train" and os.path.exists(labels_path):
            print(f'Loading {mode} features and labels from "{features_dir}"')
            return np.load(features_path), np.load(labels_path)
        elif mode == "test":
            print(f'Loading {mode} features from "{features_dir}"')
            return np.load(features_path), None

    # Read in data
    eeg1, _ = read_csv(os.path.join(data_dir, mode + SUFFIX_EEG1_DATA_CSV))
    eeg2, _ = read_csv(os.path.join(data_dir, mode + SUFFIX_EEG2_DATA_CSV))
    emg, _ = read_csv(os.path.join(data_dir, mode + SUFFIX_EMG_DATA_CSV))
    if eeg1 is None or eeg2 is None or emg is None:
        raise RuntimeError(f"There was a problem with reading the {mode} data")

    # Read in labels
    if mode == "train":
        labels, _ = read_csv(os.path.join(data_dir, TRAINING_LABELS_CSV))
        if labels is None:
            raise RuntimeError("There was a problem with reading CSV data")
        # Make classes start from 0, as they start from 1
        labels = labels[:, 1] - 1
    else:
        labels = None

    processed_data = fft(eeg1[:, 1:])
    processed_data = np.stack([processed_data.real, processed_data.imag], 1)

    eeg2_fft = fft(eeg2[:, 1:])
    processed_data = np.column_stack(
        (processed_data, np.expand_dims(eeg2_fft.real, 1), np.expand_dims(eeg2_fft.imag, 1))
    )

    emg_fft = fft(emg[:, 1:])
    processed_data = np.column_stack(
        (processed_data, np.expand_dims(emg_fft.real, 1), np.expand_dims(emg_fft.imag, 1))
    )

    np.save(features_path, processed_data)
    if labels is not None:
        np.save(labels_path, labels)

    return processed_data, labels


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
    balance_weights: bool = False,
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
        score = evaluate_model_task4(
            model,
            X_train,
            Y_train,
            k=3,
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


if __name__ == "__main__":
    parser = ArgumentParser(
        description="The entry point for the scripts of Task 4",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/task4",
        help="path to the directory containing the task data",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="config/task4/",
        help="where the extracted features are stored or should be stored",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/task4",
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
        default="config/task4.yaml",
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
        default="config/features-task4.txt",
        help="where the most optimal features are stored",
    )
    parser.add_argument("--visual", action="store_true", help="enable model visualizations")
    parser.add_argument("--augment", type=int, help="augmentation factor for the minority class")
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
        default="dist/submission4.csv",
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
    tune_parser.add_argument(
        "--extend", action="store_true", help="only tune parameters not present in config file"
    )

    __main(parser.parse_args())
