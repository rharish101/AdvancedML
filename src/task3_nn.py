"""Neural network model for task 3."""
import os
from datetime import datetime
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    CrossEntropyLoss,
    Flatten,
    Linear,
    Module,
    ReLU,
    Sequential,
    Softmax,
)
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from typings import BaseClassifier, CSVData


class ResidualBlock(Module):
    """Layer for residual block.

    The block consists of:
        * Conv
        * BatchNorm
        * ReLU
        * Conv
        * Skip connection
        * BatchNorm
        * ReLU
        * MaxPool
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        """Initialize the layers."""
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.before = Sequential(
            Conv1d(channels, channels, kernel_size=kernel_size, bias=False, padding=padding),
            BatchNorm1d(channels),
            ReLU(),
            Conv1d(channels, channels, kernel_size=kernel_size, bias=False, padding=padding),
        )
        self.after = Sequential(BatchNorm1d(channels), ReLU())

    def forward(self, x: Tensor) -> Tensor:
        """Return the output."""
        return self.after(self.before(x) + x)


class NN(BaseClassifier):
    """The neural network model for time-series classification."""

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        log_dir: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        balance_weights: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """Store hyper-params.

        Parameters
        ----------
        epochs: The max epochs to train the model
        batch_size: The batch size for training the model
        log_dir: The directory where to save TensorBoard logs
        learning_rate: The learning rate for Adam
        weight_decay: The L2 regularization parameter for Adam (using decoupled L2)
        balance_weights: Whether to use inverse class frequency weights for the loss
        random_state: The random seed for both numpy and PyTorch
        """
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.balance_weights = balance_weights
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_model(self, num_classes: int) -> None:
        self.model = Sequential(
            Conv1d(1, 32, 3, padding=1),
            ResidualBlock(32),
            ResidualBlock(32),
            Flatten(),
            Linear(32 * 180, num_classes),
        ).to(self.device)

    @staticmethod
    def _timestamp_dir(base_dir: str) -> str:
        """Add a time-stamped directory after the base directory."""
        return os.path.join(base_dir, datetime.now().isoformat())

    def _gen_batches(
        self,
        X: List[CSVData],
        y: Optional[CSVData] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Yield data elements as batches."""
        indices = np.arange(len(X))
        if rng is not None:
            rng.shuffle(indices)

        for i in range(0, len(X), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_X = torch.from_numpy(X[batch_indices]).to(self.device)
            batch_X = batch_X.unsqueeze(1)  # [n, l] => [n, 1, l]

            if y is None:
                yield batch_X
            else:
                batch_y = torch.from_numpy(y[batch_indices]).to(self.device)
                yield batch_X, batch_y

    def fit(self, X: CSVData, y: CSVData) -> None:
        """Initialize and train the model.

        Parameters
        ----------
        X: The list of 2D input data where the 1st dimension is of variable length
        y: The corresponding output integer labels
        """
        X = X.astype(np.float32)
        y = y.astype(np.int64)

        class_count = np.bincount(y.astype(np.int64))
        num_classes = len(class_count)
        self.classes_ = np.arange(num_classes)
        total_batches = np.ceil(len(X) / self.batch_size)

        self._init_model(num_classes)

        if self.balance_weights:
            class_weights = torch.from_numpy((1 / class_count).astype(np.float32)).to(self.device)
            # Normalize weights such that equal class frequencies imply 1:1:1 ratio
            class_weights /= class_weights.min()
        else:
            class_weights = None

        loss_func = CrossEntropyLoss(class_weights)
        optim = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        writer = SummaryWriter(self._timestamp_dir(self.log_dir))

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        np_rng = np.random.default_rng(self.random_state)

        self.model.train()
        for ep in range(1, self.epochs + 1):
            running_loss = 0.0

            for batch_X, batch_y in tqdm(
                self._gen_batches(X, y, np_rng),
                desc=f"Epoch {ep}/{self.epochs}",
                total=int(total_batches),
            ):
                optim.zero_grad()
                loss = loss_func(self.model(batch_X), batch_y)
                running_loss += loss.detach()
                loss.backward()
                optim.step()

            writer.add_scalar("loss", running_loss / total_batches, ep)
            for name, param in self.model.named_parameters():
                writer.add_histogram(name, param, ep)

    def predict_proba(self, X: CSVData) -> np.ndarray:
        """Predict the class probabilites.

        Parameters
        ----------
        X: The list of 2D input data where the 1st dimension is of variable length

        Returns
        -------
        The 2D array of class probabilities
        """
        X = X.astype(np.float32)
        pred = []
        softmax_func = Softmax(dim=-1)

        self.model.eval()
        with torch.no_grad():
            for batch_X in self._gen_batches(X):
                pred.append(softmax_func(self.model(batch_X)).cpu().numpy())

        return np.concatenate(pred, axis=0)

    def predict(self, X: List[CSVData]) -> np.ndarray:
        """Predict the classes.

        Parameters
        ----------
        X: The list of 2D input data where the 1st dimension is of variable length

        Returns
        -------
        The 1D array of class predictions
        """
        return self.predict_proba(X).argmax(axis=1)
