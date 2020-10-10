"""The entry point for the scripts for Task 0."""
from typing import Any, cast

import numpy as np
from pandas import DataFrame

from utilities.data import create_submission_file, print_array, print_array_statistics, read_csv

TASK_DATA_DIRECTORY = "data/task0"
TRAIN_CSV_FILE = f"{TASK_DATA_DIRECTORY}/train.csv"
TEST_CSV_FILE = f"{TASK_DATA_DIRECTORY}/test.csv"

OUTPUT_FILE = "dist/submission.csv"

# Read in data
(csv_data_train, csv_header_train) = read_csv(TRAIN_CSV_FILE)
(csv_data_test, csv_header_test) = read_csv(TEST_CSV_FILE)

if csv_data_train is None or csv_data_test is None:
    exit(1)

df_train = DataFrame(csv_data_train)
X_test = DataFrame(csv_data_test)

Y_train = df_train["y"]
X_train = df_train.drop(["y", "Id"], 1)

W = cast(Any, np.linalg).lstsq(X_train, Y_train, rcond=-1)[0]

Y_test = DataFrame(X_test.drop("Id", 1).dot(W))

Y_test.insert(0, "Id", X_test["Id"])

# Print preview of data
print_array(csv_data_train, csv_header_train or ())

# Print statistics of data
print_array_statistics(csv_data_train)

# Write results to a submission file
create_submission_file(OUTPUT_FILE, cast(Any, np).asarray(Y_test))
