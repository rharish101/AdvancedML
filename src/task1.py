"""The entry point for the scripts for Task 1."""
import numpy
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

    # We can substitute this for a more complex imputer later on
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)

    create_submission_file(OUTPUT_FILE, numpy.array([]))


def __print_data(data: CSVData, labels: CSVData, header: CSVHeader) -> None:
    # Print preview of data
    print_array(data, header)
    print_array(labels, header)

    # Print statistics of data
    print_array_statistics(data)
    print_array_statistics(labels)


__main()
