"""The entry point for the scripts for Task 1."""
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer

from typings import CSVData, CSVHeader
from utilities.data import create_submission_file, print_array, print_array_statistics, read_csv

TASK_DATA_DIRECTORY = "data/task1"
TRAINING_DATA_PATH = f"{TASK_DATA_DIRECTORY}/X_train.csv"
TRAINING_LABELS_PATH = f"{TASK_DATA_DIRECTORY}/y_train.csv"
TEST_DATA_PATH = f"{TASK_DATA_DIRECTORY}/X_test.csv"
OUTPUT_FILE = "dist/submission1.csv"


def __main():
    # Read in data
    X_train, _ = read_csv(TRAINING_DATA_PATH)
    Y_train, _ = read_csv(TRAINING_LABELS_PATH)
    X_test, _ = read_csv(TEST_DATA_PATH)

    if X_train is None or Y_train is None or X_test is None:
        raise RuntimeError("There was a problem with reading CSV data")

    # Remove training IDs, as they are in sorted order for training data
    X_train = X_train[:, 1:]
    Y_train = Y_train[:, 1:]

    # Save test IDs as we need to add them to the submission file
    test_ids = X_test[:, 0]
    X_test = X_test[:, 1:]

    # We can substitute this for a more complex imputer later on
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = xgb.XGBRegressor()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Add IDs
    submission = np.stack([test_ids, Y_pred], 1)
    create_submission_file(OUTPUT_FILE, submission)


def __print_data(data: CSVData, labels: CSVData, header: CSVHeader) -> None:
    # Print preview of data
    print_array(data, header)
    print_array(labels, header)

    # Print statistics of data
    print_array_statistics(data)
    print_array_statistics(labels)


__main()
