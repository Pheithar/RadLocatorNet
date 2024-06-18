from typing import Dict

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim


class Conv1DNetwork(L.LightningModule):
    def __init__(self):
        """TODO: Allow input to change params and not hardcode it"""
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(69, 25, 3, padding="same"),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Conv1d(25, 25, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(25 * 95, 500),
            nn.ReLU(),
            nn.Linear(500, 3),
        )
        self.loss_func = nn.MSELoss()
        self.train_loss = []
        self.val_loss = []

    def training_step(self, batch: Dict[str, np.ndarray], batch_idx: int) -> float:
        """TODO: Docstring for training_step.

        Args:
            batch (Dict[str, np.ndarray]): _description_
            batch_idx (int): _description_

        Returns:
            float: The loss of the training step
        """
        x = batch["signal"]
        y = batch["label"]

        x = self.model(x)

        loss = self.loss_func(x, y)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: Dict[str, np.ndarray], batch_idx: int) -> float:
        """TODO: Docstring for validation_step.

        Args:
            batch (Dict[str, np.ndarray]): _description_
            batch_idx (int): _description_

        Returns:
            float: _description_
        """
        x = batch["signal"]
        y = batch["label"]

        x = self.model(x)

        loss = self.loss_func(x, y)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch: Dict[str, np.ndarray], batch_idx: int) -> float:
        """


        Args:
            batch (Dict[str, np.ndarray]): _description_
            batch_idx (int): _description_

        Returns:
            float: _description_
        """

        x = batch["signal"]
        y = batch["label"]

        x = self.model(x)

        loss = self.loss_func(x, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def on_train_epoch_end(self) -> None:
        self.train_loss.append(self.trainer.callback_metrics["train_loss"].cpu())
        self.val_loss.append(self.trainer.callback_metrics["val_loss"].cpu())

    def on_train_end(self) -> None:
        plt.plot(self.train_loss, label="train loss FC")
        plt.plot(self.val_loss, label="val loss FC")
        plt.legend()
        plt.show()
