from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torchmetrics import F1Score
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as L

from loss import bceloss, cross_entropy_loss
from metric import OptimalF1Score


class Classifier(L.LightningModule):
    """PyTorch Lightning Module for Image Binary Classification.

    Args:
        net (nn.Module): PyTorch model for image classification.
        lr (float): Learning rate for the optimizer. Default: 1e-3.
        task (str): ``'binary'``, ``'multiclass'`` or ``multilabel``.
        num_classes (int): Number of output classes.
        loss_function (str): Loss function for the model.
        loss_function_kwargs (dict): Keyword argument for loss function.
        optimizer_kwargs (dict): Keyword argument for optimizer.
        scheduler_kwargs (dict): Keyword argument for scheduler.
    """

    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        task: str = "binary",
        num_classes: int = 1,
        loss_function: str = "BCE",
        loss_function_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        scheduler_kwargs: dict = dict()
    ) -> None:

        super().__init__()

        self.net = net
        self.lr = lr
        self.task: str = task
        self.num_classes: int = num_classes
        self.loss_function: Callable = self.__get_loss_function(loss_function)
        self.loss_function_kwargs: dict = loss_function_kwargs
        self.optimizer_kwargs: dict = optimizer_kwargs
        self.scheduler_kwargs: dict = scheduler_kwargs

        self.f1score = F1Score(task=self.task)
        self.optim_f1score = OptimalF1Score()

    def __get_loss_function(self, name: Literal["CE", "BCE"]) -> Callable:
        """Load a loss function for binary classification task.

        Args:
            name (Literal["CE", "BCE"]): Name of loss function to load.

        Returns:
            Callable: The requested loss function.
        """
        if name == "CE":
            loss_function = cross_entropy_loss

        if name == "BCE":
            loss_function = bceloss

        return loss_function

    def configure_optimizers(self) -> tuple:
        """Configure the optimizers and learning rate scheduler for the model.

        Returns:
            tuple: Optimizer and learning rate scheduler.
        """

        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            **self.optimizer_kwargs
        )

        scheduler = CosineAnnealingLR(optimizer, **self.scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape 
                (batch_size, input_channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.net(x)

    def predict_proba(self, x: Tensor) -> Tensor:
        """Predicts the probability of each class.

        Args:
            x (Tensor): Input tensor of shape 
                (batch_size, input_channels, height, width).

        Returns:
            Tensor: Probability of the sample for each class in the model,
        """
        logits = self(x)

        if self.task == "binary" and self.num_classes == 2:
            proba = logits.sigmoid()[:, [1]]

        else:
            proba = logits.sigmoid()

        return proba

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """Training step for a single batch.

        Args:
            batch (tuple): Input batch consisting of (x, y) 
                where x is the input tensor and y is the target tensor.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Loss value for the batch.
        """
        x, y = batch
        logits = self(x)

        loss = self.loss_function(
            logits,
            y,
            **self.loss_function_kwargs
        )

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step for a single batch.

        Args:
            batch (tuple): Input batch consisting of (x, y) 
                where x is the input tensor and y is the target tensor.
            batch_idx (int): Index of the batch.
        """
        self.evaluate(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Callback for validation epoch end."""
        self.threshold = self.optim_f1score.threshold
        self.log(
            f"threshold",
            self.optim_f1score.threshold,
            prog_bar=True
        )
        super().on_validation_epoch_end()

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Test step for a single batch.

        Args:
            batch (tuple): Input batch consisting of (x, y) 
                where x is the input tensor and y is the target tensor.
            batch_idx (int): Index of the batch.
        """
        self.evaluate(batch, "test")

    def evaluate(self, batch: tuple, stage: str = None) -> None:
        """Evaluate Model for a single batch in validation and test step.

        Args:
            batch (tuple): Input batch consisting of (x, y) 
                where x is the input tensor and y is the target tensor.
            stage (str, optional): ... Defaults to None.
        """
        x, y = batch
        logits = self(x)

        loss = self.loss_function(
            logits,
            y,
            **self.loss_function_kwargs
        )
        self.log(f"{stage}_loss", loss, prog_bar=True)

        if self.task == "binary" and self.num_classes == 2:
            proba = logits.sigmoid()[:, [1]]

        else:
            proba = logits.sigmoid()

        preds = torch.where(proba > 0.5, 1, 0)
        acc = accuracy(preds, y, self.task)

        self.f1score(proba, y)
        self.optim_f1score(proba, y)

        self.log(f"{stage}_accuracy", acc, prog_bar=True)
        self.log(
            f"{stage}_f1score",
            self.f1score,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        self.log(
            f"{stage}_optim_f1score",
            self.optim_f1score,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
