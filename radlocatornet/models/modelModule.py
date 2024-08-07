import lightning as L
from torch import nn, optim
import torch


class RadLocatorNetworkModule(L.LightningModule):
    """Model for the RadLocator Project. It requires a Python Network as input, and this class is in charge of training, evaluating and testing the model. The idea is that any PyTorch model can be used as long as it is passed as an argument to this class.

    The model passed as input does not need a forward method, as it is implemented in this class. The model should be a PyTorch model, and the loss function should be implemented in this class.

    The only necessary methods that have to come from the `model` are the initialization and the forward method. The rest of the methods are implemented in this class.

    Attributes:
        model (nn.Module): The model as a PyTorch model that will be trained
        loss_function (nn.Module): The loss function to use. It should be a PyTorch loss function
        optimizer (optim.Optimizer): The optimizer to use. It should be a PyTorch optimizer
        train_step_outputs (list[torch.Tensor]): The outputs of the model during training
        validation_step_outputs (list[torch.Tensor]): The outputs of the model during validation
        test_step_outputs (list[torch.Tensor]): The outputs of the model during testing
        train_step_targets (list[torch.Tensor]): The targets of the model during training
        validation_step_targets (list[torch.Tensor]): The targets of the model during validation
        test_step_targets (list[torch.Tensor]): The targets of the model during testing
    """

    model: nn.Module
    loss_function: nn.Module
    optimizer: optim.Optimizer

    train_step_outputs: list[torch.Tensor]
    validation_step_outputs: list[torch.Tensor]
    test_step_outputs: list[torch.Tensor]

    train_step_targets: list[torch.Tensor]
    validation_step_targets: list[torch.Tensor]
    test_step_targets: list[torch.Tensor]

    def __init__(
        self, model: nn.Module, loss_function: nn.Module, optimizer: optim.Optimizer
    ) -> None:
        """Initialize the model.

        Args:
            model (nn.Module): A PyTorch model
            loss_function (nn.Module): The loss function
            optimizer (optim.Optimizer): The optimizer
        """
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        # Keep track of the outputs and targets. They are not necessary in this code, but are useful or callbacks
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.train_step_targets = []
        self.validation_step_targets = []
        self.test_step_targets = []

        # Save the hyperparameters
        # self.save_hyperparameters()

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure the optimizer

        Returns:
            optim.Optimizer: The optimizer
        """
        return self.optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model. It should just be a call to the model

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

    def on_train_epoch_end(self) -> None:
        """As right now, this only clears the outputs and targets. This is useful for callbacks that need to keep track of the outputs and targets of the model. It is necessary to clear them at the end of the epoch, as they are lists that keep growing with each batch."""
        self.train_step_outputs.clear()
        self.train_step_targets.clear()
        # assert False, "Finished from train module"

    def on_validation_epoch_end(self) -> None:
        """As right now, this only clears the outputs and targets. This is useful for callbacks that need to keep track of the outputs and targets of the model. It is necessary to clear them at the end of the epoch, as they are lists that keep growing with each batch."""
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()

    def on_test_epoch_end(self) -> None:
        """As right now, this only clears the outputs and targets. This is useful for callbacks that need to keep track of the outputs and targets of the model. It is necessary to clear them at the end of the epoch, as they are lists that keep growing with each batch."""
        self.test_step_outputs.clear()
        self.test_step_targets.clear()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step of the model

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch. The first element is the signal, the second is the label
            batch_idx (int): The index of the batch

        Returns:
            torch.Tensor: The loss
        """
        loss, y_hat, y = self._common_step(batch)
        self.log_dict({"Train Loss": loss}, on_step=False, on_epoch=True, prog_bar=True)

        # Keep track of the outputs and targets.
        self.train_step_outputs.append(y_hat)
        self.train_step_targets.append(y)

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step of the model

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch. The first element is the signal, the second is the label
            batch_idx (int): The index of the batch

        Returns:
            torch.Tensor: The loss
        """
        loss, y_hat, y = self._common_step(batch)
        self.log_dict(
            {"Validation Loss": loss}, on_step=False, on_epoch=True, prog_bar=True
        )

        # Keep track of the outputs and targets.
        self.validation_step_outputs.append(y_hat)
        self.validation_step_targets.append(y)

        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step of the model

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch. The first element is the signal, the second is the label
            batch_idx (int): The index of the batch

        Returns:
            torch.Tensor: The loss
        """
        loss, y_hat, y = self._common_step(batch)
        self.log_dict({"Test Loss": loss}, on_step=False, on_epoch=True)

        # Keep track of the outputs and targets.
        self.test_step_outputs.append(y_hat)
        self.test_step_targets.append(y)

        return loss

    def _common_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Common step for the train test and validation steps

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch. The first element is the signal, the second is the label

        Returns:
            torch.Tensor: The loss
            torch.Tensor: The prediction
            torch.Tensor: The target
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        return loss, y_hat, y
