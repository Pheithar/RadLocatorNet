import lightning as L
import numpy as np
import torch
from torch import nn, optim

from radlocatornet.utils import get_activation_function, get_loss_function


class FullyConnectedNetwork(L.LightningModule):
    """Fully connected network for the RadLocator project. A simple fully connected network. The expected input is a 2D array with the signals flattened. The output is a 3D array with the x, y, z coordinates of the source.

    Attributes:
        model (nn.Sequential): The model
        loss_func (nn.MSELoss): The loss function
        optimizer (optim.Optimizer): The optimizer. For now defaults always to Adam.

    """

    def __init__(
        self,
        input_size: int,
        architecture: list[dict[str, str | int]],
        output_size: int,
        output_activation: str,
        loss_function: nn.Module,
        learning_rate: float = 1e-3,
    ) -> None:
        """Initialize the model. For now, the optimizers defaults to Adam, and cannot be changed. If necessary, it can be changed in the future.

        The model architecture is a list of dictionaries, where each dictionary has the following keys
        - size: The size of the layer
        - activation: The activation function
        - batch_norm: Whether to use batch normalization
        - dropout: The dropout rate. If 0, no dropout is used


        Args:
            input_size (int): Size of the input. Should be `number of signals * signal length + 3`
            architecture (list[dict[str, str  |  int]]): The architecture of the model, as a list of dictionaries, where each represent a layer
            output_size (int): The size of the output. Should be 3, at least for now
            output_activation (str): The activation function of the output
            loss_function (nn.Module): The loss function
            learning_rate (float, optional): The learning rate of the optimizer. Defaults to 1e-3.
        """
        super().__init__()

        self.model = nn.Sequential()
        in_features = input_size

        for i, layer in enumerate(architecture):
            hidden_size = layer["size"]
            self.model.add_module(
                f"linear_{i} ({in_features}x{hidden_size})",
                nn.Linear(in_features, hidden_size),
            )
            if layer["batch_norm"]:
                self.model.add_module(
                    f"batch_norm_{i}",
                    nn.BatchNorm1d(hidden_size),
                )
            self.model.add_module(
                f"activation_{i} ({layer['activation']})",
                get_activation_function(layer["activation"]),
            )
            if layer["dropout"]:
                self.model.add_module(
                    f"dropout_{i} ({layer['dropout']})",
                    nn.Dropout(layer["dropout"]),
                )
            in_features = hidden_size

        self.model.add_module(
            f"output ({in_features}x{output_size})",
            nn.Linear(in_features, output_size),
        )
        self.model.add_module(
            f"output_activation ({output_activation})",
            get_activation_function(output_activation),
        )
        self.loss_function = get_loss_function(loss_function)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def configure_optimizers(self) -> optim.Optimizer:
        """Returns the optimizer that has been set up

        Returns:
            optim.Optimizer: The optimizer
        """
        return self.optimizer

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the model. Straightforward as it is a Sequential model

        Args:
            x (np.ndarray): The input, a 2D array of shape (batch_size, input_size)

        Returns:
            np.ndarray: The output of the model
        """
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training step of the model. FOr now it does not need `y_hat` and `y`, but it is kept for future reference. Same with the `batch_idx`

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch. The first element is the signal, the second is the label
            batch_idx (int): The index of the batch

        Returns:
            torch.Tensor: The loss
        """
        loss, _, _ = self._common_step(batch)
        self.log_dict({"train_loss": loss}, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """TODO"""
        pass

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validation step of the model

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch. The first element is the signal, the second is the label
            batch_idx (int): The index of the batch

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The loss, the predicted values and the true values
        """
        loss, _, _ = self._common_step(batch)
        self.log_dict({"val_loss": loss}, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        """TODO"""
        pass

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Test step of the model

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch. The first element is the signal, the second is the label
            batch_idx (int): The index of the batch

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The loss, the predicted values and the true values
        """
        loss, _, _ = self._common_step(batch)
        self.log_dict({"test_loss": loss}, on_step=False, on_epoch=True)
        return loss

    def _common_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Common step for the train test and validation steps

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch. The first element is the signal, the second is the label

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The obtained loss, the predicted values and the true values
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        return loss, y_hat, y
