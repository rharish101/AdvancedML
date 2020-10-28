"""A simple starter neural network in PyTorch."""

import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """A simple PyTorch neural network model."""

    def __init__(self, feature_count: int):
        """Initialize the neural network.

        Parameters
        ----------
        feature_count (int): The size of the input layer (number of features)
        """
        super(SimpleNN, self).__init__()

        self.normalize = nn.BatchNorm1d(feature_count)
        self.hidden1 = nn.Linear(feature_count, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.hidden3 = nn.Linear(512, 256)
        self.hidden4 = nn.Linear(256, 128)
        self.hidden5 = nn.Linear(128, 64)
        self.hidden6 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        """Do a forward pass in the neural network.

        Parameters
        ----------
        x (Tensor): Input tensor for the neural network

        Returns
        -------
        Tensor: The output of the neural network
        """
        x = self.normalize(x)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = self.output(x)

        return x
