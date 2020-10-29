"""Contains utilities for data I/O and processing."""
from os import path
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
from pandas import DataFrame
from tabulate import tabulate
from tensorboard.plugins import projector

from typings import Array2D, CSVData, CSVHeader

LOG_DIRECTORY = "logs"


def read_csv(
    file_path: str,
    delimiter: str = ",",
    includes_header: bool = True,
) -> Tuple[Optional[CSVData], Optional[CSVHeader]]:
    """Read in data from a CSV file.

    Parameters
    ----------
    file_path (str): File to read CSV data from

    delimiter (str): The character between values in a row of CSV data

    includes_header (boolean): `True` if the first row of data contains the column names,
        `False` otherwise.

    Returns
    -------
    Tuple[Union[CSVData, None], Union[CSVHeader, None]]: The tuple of CSV data and header.
        The data is a np array with shape (number_of_rows, number_of_valued_in_a_row)
        containing data from the CSV file (without the header, if specified).
        The header is a tuple of the values in the first row of the CSV data
    """
    print(f'Loading "{file_path}"...')

    try:
        data = np.genfromtxt(file_path, delimiter=delimiter, names=includes_header)
    except FileNotFoundError:
        print(
            f"Error: could not read CSV file at {file_path} .",
            "Make sure the given path is correct and run the program again",
        )
        data = None

    header: Union[CSVHeader, None] = None
    if data is not None:
        header = data.dtype.names
        # Convert np structured array to multi-dimensional array
        data = data.view(np.float).reshape(data.shape + (-1,))

    return (data, header)


def print_array(np_array: Array2D, header: CSVHeader) -> None:
    """Print a preview (first 10 rows) of a np array.

    Parameters
    ----------
    np_array (NDArray[(Any, Any,), Any]): A 2-dimensional np array to print to the
        standard output
    """
    table = tabulate(np_array[:10, :10], list(header), tablefmt="github", floatfmt="f")

    print()
    print(table)

    more_columns_text = f" and {np_array.shape[1] - 10} columns" if np_array.shape[1] > 10 else ""

    print(
        f"---- {np_array.shape[0] - 10} rows{more_columns_text} not shown ----\n"
        if np_array.shape[0] > 10
        else ""
    )


def print_array_statistics(np_array: Array2D) -> None:
    """Print some statistics about the given data using Pandas.

    Parameters
    ----------
    np_array (NDArray[(Any, Any,), Any]): The data to print statistics about
    """
    data_frame = DataFrame(np_array)

    print()
    print(data_frame.iloc[:, :10].describe())
    print(
        f"---- {data_frame.shape[1] - 10} columns not shown (out of {data_frame.shape[1]}) ----"
        if data_frame.shape[1] > 10
        else ""
    )


def create_submission_file(
    file_path: str,
    data: CSVData,
    header: CSVHeader = ("id", "y"),
    delimiter: str = ",",
) -> None:
    """Create a submission file in CSV format.

    Parameters
    ----------
    file_path (str): Path of the file to be created

    data (CSVData): The data to dump into the created CSV file

    header (CSVHeader, optional): A tuple with the header values to be added to the top of the file.
        Defaults to ('Id', 'y').

    delimiter (str, optional): The character between values in a row of CSV data. Defaults to ','.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        file_path,
        data,
        delimiter=delimiter,
        header=str.join(",", header),
        comments="",
    )


def visualize_data(data: CSVData, ids: List[str], name: str, log_directory: str = LOG_DIRECTORY):
    """Visualize the given data by creating files for the Tensorboard projector.

    Creates `.tsv` and `.pbtxt` files in the log directory. To use it in Tensorboard,
    run "tensorboard --logdir <log directory path>" and choose PROJECTOR from the dropdown list."

    Parameters
    ----------
    data (CSVData): np array containing the data to be visualized

    ids (List[str]): ID for each vector contained in the data array
        (i.e. `len(ids) == `data.shape[0]`)

    name (str): Name for the dataset contained in the data array ()

    log_directory (str, optional): Directory name to store the created Tensorboard logfiles in.
        Defaults to LOG_DIRECTORY
    """
    Path(log_directory).mkdir(parents=True, exist_ok=True)

    if " " in name:
        print("Visualization error: Please specify a name without whitespaces")
        return

    data_file_name = f"{name}.tsv"
    metadata_file_name = f"{name}_metadata.tsv"

    np.savetxt(path.join(log_directory, data_file_name), data, delimiter="\t", fmt="%f")

    with open(path.join(log_directory, metadata_file_name), "w") as metadata_writer:
        for data_id in ids:
            metadata_writer.write(f"{data_id}\n")

    config = projector.ProjectorConfig()
    embedding = cast(Any, config).embeddings.add()
    embedding.tensor_path = data_file_name
    embedding.metadata_path = metadata_file_name
    embedding.tensor_name = name

    projector.visualize_embeddings(log_directory, config)
    print(
        f'Run "tensorboard --logdir {log_directory}" and choose PROJECTOR',
        "to see the data visualization\n",
    )


def run_data_diagnostics(data: CSVData, labels: CSVData, header: CSVHeader) -> None:
    """Run basic diagnorstics on the given data.

    Parameters
    ----------
    data (CSVData): Data to run diagnostics on

    labels (CSVData): Labels corresponding the the given data in the `data` parameter

    header (CSVHeader): Headers corresponding to the given data in the `data`parameter
    """
    # Print preview of data
    print_array(data, header)
    print_array(labels, header)

    # Print statistics of data
    print_array_statistics(data)
    print_array_statistics(labels)

    # Tensorboard will stuck on "computing PCA" if there are non-number values in the array
    data = np.nan_to_num(data)

    # Create a TensorBoard projector to visualize data
    visualize_data(data[:, 1:], data[:, 0].astype(int), "input_data")
