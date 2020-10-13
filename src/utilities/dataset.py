"""Contains a Dataset utility."""

from typing import Any, Tuple

import numpy
import torch
from torch.utils.data import Dataset

from typings import CSVData


class DataSet(Dataset):
    """Task dataset."""

    def __init__(self, csv_data: CSVData, csv_labels):
        """Create a new dataset from the given data.

        Parameters
        ----------
        csv_data (CSVData): The data as a numpy array coming from a CSV file
        """
        self.data = csv_data.astype(numpy.float32)
        self.labels = csv_labels.astype(numpy.float32)

    def __len__(self) -> int:
        """Get the size of the dataset.

        Returns
        -------
        (int): Size of the dataset
        """
        return self.data.shape[0]

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx: The ids of the desired elements

        Returns
        -------
        (Tuple): Tuple of the data element and its corresponding label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx], self.labels[idx]
