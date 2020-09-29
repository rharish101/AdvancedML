"""Contains utilities for data I/O and processing."""

from pathlib import Path
from typing import Any, Tuple, Union, cast

import numpy
from pandas import DataFrame
from tabulate import tabulate

from typings import Array2D, CSVData, CSVHeader


def read_csv(
    file_path: str,
    delimiter: str = ",",
    includes_header: bool = True,
) -> Tuple[Union[CSVData, None], Union[CSVHeader, None]]:
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
        The data is a numpy array with shape (number_of_rows, number_of_valued_in_a_row)
        containing data from the CSV file (without the header, if specified).
        The header is a tuple of the values in the first row of the CSV data
    """
    try:
        data = cast(
            CSVData,
            numpy.genfromtxt(file_path, delimiter=delimiter, dtype=None, names=includes_header),
        )
    except FileNotFoundError:
        print(
            f"Error: could not read CSV file at {file_path} . ",
            "Make sure the given path is correct and run the program again",
        )
        data = None

    header: Union[CSVHeader, None] = None
    if data is not None:
        header = data.dtype.names

    return (data, header)


def print_array(numpy_array: Array2D) -> None:
    """Print a preview (first 10 rows) of a numpy array.

    Parameters
    ----------
    numpy_array (NDArray[(Any, Any,), Any]): A 2-dimensional numpy array to print to the
        standard output
    """
    table = tabulate(
        numpy_array[:10],
        list(numpy_array.dtype.names),
        tablefmt="github",
    )

    print()
    print(table)

    if numpy_array.shape[0] > 10:
        print(f"{numpy_array.shape[0] - 10} more rows...")

    print()


def print_array_statistics(numpy_array: Array2D) -> None:
    """Print some statistics about the given data using Pandas.

    Parameters
    ----------
    numpy_array (NDArray[(Any, Any,), Any]): The data to print statistics about
    """
    print()
    print(DataFrame(numpy_array).describe())
    print()


def create_submission_file(
    file_path: str,
    data: CSVData,
    header: CSVHeader = ("Id", "y"),
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

    cast(Any, numpy).savetxt(
        file_path,
        data,
        delimiter=delimiter,
        header=str.join(",", header),
        comments="",
    )
