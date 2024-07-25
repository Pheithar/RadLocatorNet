from lightning import Callback, LightningModule, Trainer
import torch
from radlocatornet.models import RadLocatorNetworkModule
from radlocatornet.utils import training_plotting
import os


class ScaledRMSECallback(Callback):
    """Callback that scales the MSE by the 3D dimensions of the sensor, to return a more interpretable loss. The error unit is mm^2 because the input is in mm. After being applied the squared root, the error is in mm.

    These callbacks are to be used with the RadLocatorNet Model Module, as they need `train_step_outputs`, `evaluation_step_outputs` and `test_step_outputs` to be present in the model.
    Same as `train_step_targets`, `evaluation_step_targets` and `test_step_targets`.

    The callback is called at the end of the epoch, and at the end of the validation, train and test steps.

    Attributes:
        scale (torch.Tensor): The scale to apply to the RMSE
        progress_bar (bool): Whether to show the progress bar
        plot (bool): Whether to plot the results at the end of the training
        train_rmse (list[float]): The RMSE during training
        validation_rmse (list[float]): The RMSE during validation
        save_path (str): The path to save the plot if `plot` is True. Otherwise, it is not used
    """

    scale: torch.Tensor
    progress_bar: bool
    plot: bool
    train_rmse: list[float]
    validation_rmse: list[float]
    save_path: str

    def __init__(
        self,
        scale: tuple[float, float, float],
        progress_bar: bool = True,
        plot: bool = False,
        save_path: os.PathLike = os.getcwd(),
    ):
        """Initialize the callback

        Args:
            scale (tuple[float, float, float]): The scale to apply to the RMSE
            progress_bar (bool, optional): Whether to show the progress bar. Defaults to True.
            plot (bool, optional): Whether to plot the results at the end of the training. Defaults to False.
            save_path (os.PathLike, optional): The path to save the plot if `plot` is True. Defaults to os.getcwd().
        """
        self.scale = torch.tensor(scale)
        self.progress_bar = progress_bar
        self.plot = plot

        self.train_rmse = []
        self.validation_rmse = []

        self.save_path = save_path

    def on_train_epoch_end(
        self, trainer: Trainer, radlocatormodule: RadLocatorNetworkModule
    ) -> None:
        """Compute the scaled RMSE at the end of the epoch

        Args:
            trainer (Trainer): The trainer
            radlocatormodule (RadLocatorNetworkModule): The model
        """
        output = torch.cat(radlocatormodule.train_step_outputs)
        target = torch.cat(radlocatormodule.train_step_targets)
        output = output * self.scale.to(output.device)
        target = target * self.scale.to(target.device)

        error = torch.sqrt(((output - target) ** 2).mean())

        radlocatormodule.log(
            "train_scaled_rmse", error, on_epoch=True, prog_bar=self.progress_bar
        )
        if self.plot and not trainer.sanity_checking:
            self.train_rmse.append(error.item())

    def on_validation_epoch_end(
        self, trainer: Trainer, radlocatormodule: RadLocatorNetworkModule
    ) -> None:
        """Compute the scaled RMSE at the end of the validation

        Args:
            trainer (Trainer): The trainer
            radlocatormodule (RadLocatorNetworkModule): The model
        """
        output = torch.cat(radlocatormodule.validation_step_outputs)
        target = torch.cat(radlocatormodule.validation_step_targets)
        output = output * self.scale.to(output.device)
        target = target * self.scale.to(target.device)

        error = torch.sqrt(((output - target) ** 2).mean())

        radlocatormodule.log(
            "validation_scaled_rmse", error, on_epoch=True, prog_bar=self.progress_bar
        )

        if self.plot and not trainer.sanity_checking:
            self.validation_rmse.append(error.item())

    def on_test_epoch_end(
        self, trainer: Trainer, radlocatormodule: RadLocatorNetworkModule
    ) -> None:
        """Compute the scaled RMSE at the end of the test

        Args:
            trainer (Trainer): The trainer
            radlocatormodule (RadLocatorNetworkModule): The model
        """
        output = torch.cat(radlocatormodule.test_step_outputs)
        target = torch.cat(radlocatormodule.test_step_targets)
        output = output * self.scale.to(output.device)
        target = target * self.scale.to(target.device)

        error = torch.sqrt(((output - target) ** 2).mean())

        radlocatormodule.log(
            "test_scaled_rmse", error, on_epoch=True, prog_bar=self.progress_bar
        )

    def on_fit_end(self, trainer: Trainer, pl_module: RadLocatorNetworkModule) -> None:
        """Plot the training and validation RMSE at the end of the training if `plot` is True

        Args:
            trainer (Trainer): The trainer (not used)
            pl_module (RadLocatorNetworkModule): The model (not used)
        """
        if self.plot:
            training_plotting(
                self.train_rmse,
                self.validation_rmse,
                "Training and Validation RMSE",
                "mm",
                self.save_path,
            )
