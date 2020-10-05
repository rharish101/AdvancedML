"""The entry point for the scripts for Task 1."""
import numpy

from typings import CSVData, CSVHeader
from utilities.data import create_submission_file, print_array, print_array_statistics, read_csv

TASK_DATA_DIRECTORY = "data/task1"
TRAINING_DATA_PATH = f"{TASK_DATA_DIRECTORY}/X_train.csv"
TRAINING_LABELS_PATH = f"{TASK_DATA_DIRECTORY}/y_train.csv"
TEST_DATA_PATH = f"{TASK_DATA_DIRECTORY}/X_test.csv"

OUTPUT_FILE = "dist/submission1.csv"


def __main():
    # Read in data
    (training_data, training_header) = read_csv(TRAINING_DATA_PATH)
    (training_labels, _) = read_csv(TRAINING_LABELS_PATH)
    (test_data, _) = read_csv(TEST_DATA_PATH)

    if training_data is None or training_labels is None or test_data is None:
        print("There was a problem with reading CSV data. Exiting...")
        exit(1)

    # Comment out line below to remove diagnostic data printing
    __print_data(training_data, training_labels, training_header or ())

    # Write results to a submission file
    create_submission_file(OUTPUT_FILE, numpy.array([]))


def __print_data(data: CSVData, labels: CSVData, header: CSVHeader):
    # Print preview of data
    print_array(data, header)
    print_array(labels, header)

    # Print statistics of data
    print_array_statistics(data)
    print_array_statistics(labels)


__main()
