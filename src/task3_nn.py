"""Neural network model for task 3."""
import os
from datetime import datetime
from typing import Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import (
    GRU,
    BatchNorm1d,
    Conv1d,
    CrossEntropyLoss,
    Linear,
    Module,
    ReLU,
    Sequential,
    Softmax,
)
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from typings import BaseClassifier, CSVData


class Lambda(Module):
    """Layer for custom functions."""

    def __init__(self, func: Callable[[Tensor], Tensor]):
        """Store the function."""
        super().__init__()
        self.func = func

    def forward(self, x: Tensor) -> Tensor:
        """Return the function output."""
        return self.func(x)


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

    def _init_model_optim(self, in_channels: int, num_classes: int) -> None:
        self.model = Sequential(
            Lambda(lambda x: x.permute(1, 2, 0)),  # [l, n, c] => [n, c, l]
            Conv1d(in_channels, 128, 3, bias=False),
            BatchNorm1d(128),
            ReLU(),
            Lambda(lambda x: x.permute(2, 0, 1)),  # [n, c, l] => [l, n, c]
            GRU(128, 64),
            Lambda(lambda x: x[0][-1]),  # [l, n, c] => [n, c]
            Linear(64, num_classes),
        ).to(self.device)

    @staticmethod
    def _timestamp_dir(base_dir: str) -> str:
        """Add a time-stamped directory after the base directory."""
        return os.path.join(base_dir, datetime.now().isoformat())

    def _gen_batches(
        self, rng: np.random.Generator, X: List[CSVData], y: Optional[CSVData] = None
    ) -> Iterator[Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Yield data elements as batches."""
        indices = np.arange(len(X))
        rng.shuffle(indices)

        for i in range(0, len(X), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_X = pad_sequence(
                [torch.from_numpy(X[i].astype(np.float32)) for i in batch_indices]
            ).to(self.device)

            if y is None:
                yield batch_X
            else:
                batch_y = torch.from_numpy(y[batch_indices]).to(self.device)
                yield batch_X, batch_y

    def fit(self, X: List[CSVData], y: CSVData) -> None:
        """Initialize and train the model.

        Parameters
        ----------
        X: The list of 2D input data where the 1st dimension is of variable length
        y: The corresponding output integer labels
        """
        y = y.astype(np.int64)
        class_count = np.bincount(y.astype(np.int64))

        in_channels = X[0].shape[1]
        num_classes = len(class_count)
        self.classes_ = np.arange(num_classes)
        total_batches = np.ceil(len(X) / self.batch_size)

        self._init_model_optim(in_channels, num_classes)

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
                self._gen_batches(np_rng, X, y),
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

    def predict_proba(self, X: List[CSVData]) -> np.ndarray:
        """Predict the class probabilites.

        Parameters
        ----------
        X: The list of 2D input data where the 1st dimension is of variable length

        Returns
        -------
        The 2D array of class probabilities
        """
        pred = []
        softmax_func = Softmax(dim=-1)
        np_rng = np.random.default_rng(None)

        self.model.eval()
        with torch.no_grad():
            for batch_X in self._gen_batches(np_rng, X):
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
