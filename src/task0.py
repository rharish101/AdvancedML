"""The entry point for the scripts for Task 0."""

from utilities.data import create_submission_file, print_array, print_array_statistics, read_csv

TASK_DATA_DIRECTORY = "data/task0"
TRAIN_CSV_FILE = f"{TASK_DATA_DIRECTORY}/train.csv"
TEST_CSV_FILE = f"{TASK_DATA_DIRECTORY}/test.csv"

OUTPUT_FILE = "dist/submission.csv"

# Read in data
(csv_data, csv_header) = read_csv(TRAIN_CSV_FILE)

if csv_data is None:
    exit(1)

# Print preview of data
print_array(csv_data)

# Print statistics of data
print_array_statistics(csv_data)

# Write results to a submission file
create_submission_file(OUTPUT_FILE, csv_data)
