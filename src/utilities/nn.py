"""Contains utilities for neural network operations."""

from datetime import datetime
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine
from ignite.contrib.metrics.regression import R2Score
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Loss
from torch.utils.data import DataLoader

from models.simple_nn import SimpleNN
from utilities.dataset import DataSet


def evaluate_nn_model(X_train, Y_train):
    """Evaluate SimpleNN on the given training data.

    Parameters
    ----------
    X_train (numpy.ndarray): Training data
    Y_train : (numpy.ndarray): Training labels

    Returns
    -------
    (SimpleNN): The trained and evaluated SimpleNN neural network model
    """
    model = SimpleNN(X_train.shape[1])

    training_indices = np.random.choice(
        X_train.shape[0], int(X_train.shape[0] * 0.2), replace=False
    )

    training_loader = DataLoader(
        DataSet(X_train[training_indices], Y_train[training_indices]),
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    mask = np.ones(X_train.shape[0], dtype=bool)
    mask[training_indices] = False

    validation_loader = DataLoader(
        DataSet(X_train[mask], Y_train[mask]),
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    train_network(model, training_loader, validation_loader)

    return model


def train_network(model: nn.Module, training_loader: DataLoader, validation_loader: DataLoader):
    """Trains the given neural network model.

    Parameters
    ----------
    model (nn.Module): The PyTorch model to be trained

    training_loader (DataLoader): Training data loader

    validation_loader (DataLoader): Validation data loader
    """
    device = "cuda:0" if cast(Any, torch).cuda.is_available() else "cpu"

    if device == "cuda:0":
        model.cuda()

    optimizer = cast(Any, torch).optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    save_handler = Checkpoint(
        {"model": model, "optimizer": optimizer, "trainer": trainer},
        DiskSaver("dist/models", create_dir=True),
        n_saved=2,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=100), save_handler)

    # Create a logger
    tb_logger = TensorboardLogger(
        log_dir="logs/training" + datetime.now().strftime("-%Y%m%d-%H%M%S"), flush_secs=1
    )

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"loss": loss},
    )

    # Training evaluator
    training_evaluator = create_supervised_evaluator(
        model, metrics={"r2": R2Score(), "MSELoss": Loss(criterion)}, device=device
    )

    tb_logger.attach_output_handler(
        training_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=["MSELoss", "r2"],
        global_step_transform=global_step_from_engine(trainer),
    )

    # Validation evaluator
    evaluator = create_supervised_evaluator(
        model, metrics={"r2": R2Score(), "MSELoss": Loss(criterion)}, device=device
    )

    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["MSELoss", "r2"],
        global_step_transform=global_step_from_engine(trainer),
    )

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def log_training_results(trainer):
        training_evaluator.run(training_loader)

        metrics = training_evaluator.state.metrics
        print(
            f"Training Results - Epoch: {trainer.state.epoch}",
            f" Avg r2: {metrics['r2']:.2f} Avg loss: {metrics['MSELoss']:.2f}",
        )

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def log_validation_results(trainer):
        evaluator.run(validation_loader)

        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {trainer.state.epoch}",
            f" Avg r2: {metrics['r2']:.2f} Avg loss: {metrics['MSELoss']:.2f}\n",
        )

    trainer.run(training_loader, max_epochs=int(1e6))
