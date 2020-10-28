#!/usr/bin/env python
"""The entry point for the scripts for Task 2."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Dict, Union

import torch
import yaml
from hyperopt import fmin, hp, tpe
from typing_extensions import Final

from utilities.data import preprocess, read_csv, run_data_diagnostics
from utilities.model import (
    choose_model,
    evaluate_model,
    feature_selection,
    finalize_model,
    read_selected_features,
)
from utilities.nn import evaluate_nn_model

TASK_DATA_DIRECTORY: Final[str] = "data/task1"
TRAINING_DATA_NAME: Final[str] = "X_train.csv"
TRAINING_LABELS_NAME: Final[str] = "y_train.csv"
TEST_DATA_PATH: Final[str] = "X_test.csv"

# Search distributions for hyper-parameters
SPACE: Final = {
    "n_neighbors": hp.choice("n_neighbors", range(1, 30)),
    "contamination": hp.quniform("contamination", 0.05, 0.5, 0.05),
    "n_estimators": hp.choice("n_estimators", range(10, 500)),
    "max_depth": hp.choice("max_depth", range(2, 20)),
    "learning_rate": hp.quniform("learning_rate", 0.01, 1.0, 0.01),
    "gamma": hp.uniform("gamma", 0.0, 1.0),
    "min_child_weight": hp.uniform("min_child_weight", 1, 8),
    "subsample": hp.uniform("subsample", 0.8, 1),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1.0, 0.05),
    "reg_lambda": hp.lognormal("reg_lambda", 1.0, 1.0),
}

torch.device("cuda:0")


def __main(args: Namespace) -> None:
    # Read in data
    X_train, X_header = read_csv(f"{args.data_dir}/{TRAINING_DATA_NAME}")
    Y_train, _ = read_csv(f"{args.data_dir}/{TRAINING_LABELS_NAME}")

    if X_train is None or Y_train is None:
        raise RuntimeError("There was a problem with reading CSV data")

    if args.diagnose:
        run_data_diagnostics(X_train, Y_train, header=X_header or ())

    if args.mode == "tune":
        selected_features = read_selected_features(args.features, X_train.shape[1])

        def objective(config: Dict[str, Union[float, int]]) -> float:
            model = choose_model("xgb", **config)  # type:ignore
            X_train_new, Y_train_new, _ = preprocess(X_train, Y_train, **config)  # type:ignore
            X_train_new = X_train_new[:, selected_features]
            # Keep k low for faster evaluation
            score = evaluate_model(model, X_train_new, Y_train_new, k=5)
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

    if args.model == "nn":
        # TODO: This should be removed after the NN model is complete
        evaluate_nn_model(X_train, Y_train)
    else:
        model = choose_model(args.model, **config)

    # Preprocess before feature selection
    X_train, Y_train, imputer = preprocess(X_train, Y_train, **config)

    if args.mode == "eval" and args.train_features:
        print("Training feature selection model")
        selected_features = feature_selection(
            model, X_train, Y_train, args.cross_val, args.features
        )
    else:
        selected_features = read_selected_features(args.features, X_train.shape[1])
    X_train = X_train[:, selected_features]

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
        X_test = X_test[:, selected_features]

        finalize_model(model, X_train, Y_train, X_test, test_ids, args.output)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="The entry point for the scripts for Task 2",
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
        "--model",
        choices=["xgb", "nn"],
        default="xgb",
        help="the choice of model to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/task1.yaml",
        help="the path to the YAML config for the hyper-parameters",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="config/features-task1.txt",
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
        default=10,
        help="the k for k-fold cross-validation",
    )
    eval_parser.add_argument(
        "--train-features",
        action="store_true",
        help="whether to train the most optimal features",
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
        default="dist/submission1.csv",
        help="the path by which to save the output CSV (only used in the 'final' mode)",
    )

    # Sub-parser for hyper-param tuning
    tune_parser = subparsers.add_parser(
        "tune", description="hyper-parameter tuning", formatter_class=ArgumentDefaultsHelpFormatter
    )
    tune_parser.add_argument(
        "--max-evals",
        type=int,
        default=200,
        help="max. evaluations for hyper-opt",
    )

    __main(parser.parse_args())
