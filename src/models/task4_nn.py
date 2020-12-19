"""Neural network model for task 4."""
import os
from datetime import datetime
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import (
    ELU,
    BatchNorm1d,
    Conv1d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Linear,
    MaxPool1d,
    Module,
    Sequential,
    Softmax,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from typings import BaseClassifier, CSVData


class ResidualBlock(Module):
    """Layer for residual block.

    The block consists of:
        * Dropout
        * Conv
        * BatchNorm
        * ELU
        * Dropout
        * Conv
        * Skip connection
        * BatchNorm
        * ELU
        * MaxPool
    """

    def __init__(self, channels: int, kernel_size: int = 5, stride: int = 2, dropout: float = 0.1):
        """Initialize the layers."""
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.before = Sequential(
            Dropout(dropout),
            Conv1d(channels, channels, kernel_size=kernel_size, bias=False, padding=padding),
            BatchNorm1d(channels),
            ELU(),
            Dropout(dropout),
            Conv1d(channels, channels, kernel_size=kernel_size, bias=False, padding=padding),
        )
        self.after = Sequential(
            BatchNorm1d(channels),
            ELU(),
            MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding),
        )

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
        lr_step: int = 10000,
        lr_decay: float = 0.75,
        weight_decay: float = 0.0,
        balance_weights: bool = True,
        log_steps: int = 100,
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
        log_steps: The interval for logging into TensorBoard
        random_state: The random seed for both numpy and PyTorch
        """
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.balance_weights = balance_weights
        self.log_steps = log_steps
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_model(self, in_channels, num_classes: int) -> None:
        # Initial seq. length: 512
        self.model = Sequential(
            Conv1d(in_channels, 64, kernel_size=5, bias=False),
            BatchNorm1d(64),
            ResidualBlock(64),  # 512 => 256
            ResidualBlock(64),  # 256 => 128
            ResidualBlock(64),  # 128 => 64
            ResidualBlock(64),  # 64 => 32
            Flatten(),
            Dropout(0.3),
            Linear(64 * 32, 64, bias=False),
            BatchNorm1d(64),
            ELU(),
            Dropout(0.3),
            Linear(64, num_classes),
        ).to(self.device)

    @staticmethod
    def _timestamp_dir(base_dir: str) -> str:
        """Add a time-stamped directory after the base directory."""
        return os.path.join(base_dir, datetime.now().isoformat().replace(":", ""))

    def _gen_batches(
        self,
        X: CSVData,
        y: Optional[CSVData] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Yield data elements as batches."""
        indices = np.arange(len(X))
        if rng is not None:
            rng.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_X = torch.from_numpy(X[batch_indices]).to(self.device)

            if y is None:
                yield batch_X
            else:
                batch_y = torch.from_numpy(y[batch_indices]).to(self.device)
                yield batch_X, batch_y

    def fit(self, X: CSVData, y: CSVData) -> None:
        """Initialize and train the model.

        Parameters
        ----------
        X: The 3D input data
        y: The corresponding output integer labels
        """
        X = X.astype(np.float32)
        y = y.astype(np.int64)

        class_count = np.bincount(y.astype(np.int64))
        num_classes = len(class_count)
        in_channels = X.shape[1]
        self.classes_ = np.arange(num_classes)
        total_batches = int(np.ceil(len(X) / self.batch_size))

        self._init_model(in_channels, num_classes)

        class_weights: Optional[Tensor] = None
        if self.balance_weights:
            class_weights = torch.from_numpy((1 / class_count).astype(np.float32)).to(self.device)
            # Normalize weights such that equal class frequencies imply 1:1:1 ratio
            class_weights /= class_weights.min()

        loss_func = CrossEntropyLoss(class_weights)
        optim = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = StepLR(optim, step_size=self.lr_step, gamma=self.lr_decay)
        writer = SummaryWriter(self._timestamp_dir(self.log_dir))

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        np_rng = np.random.default_rng(self.random_state)

        self.model.train()
        global_step = 0
        for ep in range(1, self.epochs + 1):
            running_loss = 0.0

            for i, (batch_X, batch_y) in tqdm(
                enumerate(self._gen_batches(X, y, np_rng), 1),
                desc=f"Epoch {ep}/{self.epochs}",
                total=total_batches,
            ):
                optim.zero_grad()
                loss = loss_func(self.model(batch_X), batch_y)
                running_loss += loss.detach()
                loss.backward()
                optim.step()
                scheduler.step()

                if global_step % self.log_steps == 0:
                    writer.add_scalar("loss", running_loss / i, global_step)
                    for name, param in self.model.named_parameters():
                        writer.add_histogram(name, param, global_step)

                global_step += 1

    def predict_proba(self, X: CSVData) -> np.ndarray:
        """Predict the class probabilites.

        Parameters
        ----------
        X: The 3D input data

        Returns
        -------
        The 2D array of class probabilities
        """
        X = X.astype(np.float32)

        pred = []
        softmax_func = Softmax(dim=-1)
        total_batches = int(np.ceil(len(X) / self.batch_size))

        self.model.eval()
        with torch.no_grad():
            for batch_X in tqdm(self._gen_batches(X), desc="Prediction", total=total_batches):
                pred.append(softmax_func(self.model(batch_X)).cpu().numpy())

        return np.concatenate(pred, axis=0)

    def predict(self, X: CSVData) -> np.ndarray:
        """Predict the classes.

        Parameters
        ----------
        X: The 3D input data

        Returns
        -------
        The 1D array of class predictions
        """
        return self.predict_proba(X).argmax(axis=1)
