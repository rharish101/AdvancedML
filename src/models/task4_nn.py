"""Neural network model for task 4."""
import os
from datetime import datetime
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import (
    AvgPool1d,
    BatchNorm1d,
    Conv1d,
    CrossEntropyLoss,
    Dropout,
    Linear,
    Module,
    ReLU,
    Sequential,
    Softmax,
    TransformerEncoderLayer,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from typings import BaseClassifier, CSVData


class WindowBlock(Module):
    """Block for the window feature learning.

    The block consists of:
        * (Conv + BatchNorm + ReLU) x5
        * AvgPool
    """

    def __init__(self, in_channels: int):
        """Initialize the layers."""
        super().__init__()
        self.layers = Sequential(
            self._get_conv_block(in_channels, 64, kernel_size=5),  # 512 => 256
            self._get_conv_block(64, 64, kernel_size=5),  # 256 => 128
            self._get_conv_block(64, 128),  # 128 => 64
            self._get_conv_block(128, 128, stride=1),  # 64 => 64
            self._get_conv_block(128, 256, stride=1),  # 64 => 64
            AvgPool1d(kernel_size=2),  # 64 => 32
        )

    @staticmethod
    def _get_conv_block(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        padding = (kernel_size - 1) // 2
        return Sequential(
            Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                padding=padding,
            ),
            BatchNorm1d(out_channels),
            ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the output."""
        # NxCxL => LxNxC
        return self.layers(x).movedim(2, 0)


class PositionalEncoding(Module):
    """Positional encoding layer.

    This is taken from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    """

    def __init__(self, d_model, dropout=0.1, max_len=32):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Return the output."""
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class EpochBlock(Module):
    """Block for intra/inter-epoch feature learning.

    The block consists of:
        * PositionalEmbedding
        * TransformerEncoderLayer x2
        * GlobalAvgPool
    """

    def __init__(self, channels: int):
        """Initialize the layers."""
        super().__init__()
        self.layers = Sequential(
            PositionalEncoding(channels),
            TransformerEncoderLayer(channels, 1, dim_feedforward=channels),
            TransformerEncoderLayer(channels, 1, dim_feedforward=channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the output."""
        x = self.layers(x)
        x = x.mean(0)  # GlobalAvgPool over the time axis (LxNxC)
        return x


class Model(Module):
    """The combined model for sleep classification.

    This model is inspired by: https://www.mdpi.com/1660-4601/17/11/4152
    """

    def __init__(self, in_channels: int, num_classes: int):
        """Initialize the layers."""
        super().__init__()
        self.intra_block = Sequential(WindowBlock(in_channels), EpochBlock(256))
        self.inter_block = Sequential(
            EpochBlock(256),
            Linear(256, 256),
            ReLU(),
            Dropout(0.5),
            Linear(256, num_classes),
        )

    def forward(self, xs: List[Tensor]) -> Tensor:
        """Return the output."""
        # Concatenate along the batch axis for efficiency
        x = torch.cat(xs, 0)
        x = self.intra_block(x)
        # Assuming all elements of xs are the same shape
        x = x.reshape(len(xs), -1, *list(x.shape[1:]))  # (3*N)xC => 3xNxC
        x = self.inter_block(x)
        return x


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

    def _init_model(self, in_channels: int, num_classes: int) -> None:
        # Initial seq. length: 512
        self.model = Model(in_channels, num_classes).to(self.device)

    @staticmethod
    def _timestamp_dir(base_dir: str) -> str:
        """Add a time-stamped directory after the base directory."""
        return os.path.join(base_dir, datetime.now().isoformat().replace(":", ""))

    def _gen_batches_subject(
        self,
        X: CSVData,
        y: Optional[CSVData] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Yield data elements for one subject as batches."""
        indices = np.arange(len(X))
        if rng is not None:
            rng.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_idx_curr = indices[i : i + self.batch_size]
            batch_idx_prev = np.maximum(batch_idx_curr - 1, 0)
            batch_idx_next = np.minimum(batch_idx_curr + 1, len(indices) - 1)

            batch_X_prev = torch.from_numpy(X[batch_idx_prev]).to(self.device)
            batch_X_curr = torch.from_numpy(X[batch_idx_curr]).to(self.device)
            batch_X_next = torch.from_numpy(X[batch_idx_next]).to(self.device)
            batch_X = [batch_X_prev, batch_X_curr, batch_X_next]

            if y is None:
                yield batch_X
            else:
                batch_y = torch.from_numpy(y[batch_idx_curr]).to(self.device)
                yield batch_X, batch_y

    def _gen_batches(
        self,
        X: CSVData,
        y: Optional[CSVData] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Yield data elements as batches."""
        if y is None:
            y = [None] * len(X)
        else:
            y = y.reshape(len(X), -1)

        for Xi, yi in zip(X, y):
            for data in self._gen_batches_subject(Xi, yi, rng):
                yield data

    def _get_total_batches(self, input_shape: Tuple[int, int, int, int]) -> int:
        """Get the total number of batches that will be generated for this input."""
        return int(np.ceil(input_shape[1] / self.batch_size)) * input_shape[0]

    def fit(self, X: CSVData, y: CSVData) -> None:
        """Initialize and train the model.

        Parameters
        ----------
        X: The 4D SxNxCxL input data of per-subject 3D NxCxL data
        y: The corresponding 1D output integer labels
        """
        X = X.astype(np.float32)
        y = y.astype(np.int64)

        class_count = np.bincount(y.astype(np.int64))
        num_classes = len(class_count)
        in_channels = X.shape[2]
        self.classes_ = np.arange(num_classes)
        total_batches = self._get_total_batches(X.shape)

        self._init_model(in_channels, num_classes)

        class_weights: Optional[Tensor] = None
        if self.balance_weights:
            class_weights_np = len(X) / (num_classes * class_count)
            class_weights = torch.from_numpy(class_weights_np.astype(np.float32)).to(self.device)

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
        X: The 4D SxNxCxL input data of per-subject 3D NxCxL data

        Returns
        -------
        The 2D array of class probabilities
        """
        X = X.astype(np.float32)

        pred = []
        softmax_func = Softmax(dim=-1)
        total_batches = self._get_total_batches(X.shape)

        self.model.eval()
        with torch.no_grad():
            for batch_X in tqdm(self._gen_batches(X), desc="Prediction", total=total_batches):
                pred.append(softmax_func(self.model(batch_X)).cpu().numpy())

        return np.concatenate(pred, axis=0)

    def predict(self, X: CSVData) -> np.ndarray:
        """Predict the classes.

        Parameters
        ----------
        X: The 4D SxNxCxL input data of per-subject 3D NxCxL data

        Returns
        -------
        The 1D array of class predictions
        """
        return self.predict_proba(X).argmax(axis=1)
