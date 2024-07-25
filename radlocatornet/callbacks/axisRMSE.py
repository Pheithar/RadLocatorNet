from lightning import Callback, Trainer
import torch
from radlocatornet.models import RadLocatorNetworkModule
from radlocatornet.utils import training_plot_axis
import os


class AxisRMSECallback(Callback):
    """Callback that scales and calculates the error in each axis separately. The callback is called at the end of the epoch, and at the end of the validation, train and test steps. The units of the RMSE are the same as the units of the target (mm).

    These callbacks are to be used with the RadLocatorNet Model Module, as they need `train_step_outputs`, `evaluation_step_outputs` and `test_step_outputs` to be present in the model.
    Same as `train_step_targets`, `evaluation_step_targets` and `test_step_targets`.

    The callback is called at the end of the epoch, and at the end of the validation, train and test steps.

    Attributes:
        scale (torch.Tensor): The scale to apply to the MSE
        progress_bar (bool): Whether to show the progress bar
        plot (bool): Whether to plot the results at the end of the training
        train_rmse_x (list[float]): The RMSE during training for the x-axis
        train_rmse_y (list[float]): The RMSE during training for the y-axis
        train_rmse_z (list[float]): The RMSE during training for the z-axis
        validation_rmse_x (list[float]): The RMSE during validation for the x-axis
        validation_rmse_y (list[float]): The RMSE during validation for the y-axis
        validation_rmse_z (list[float]): The RMSE during validation for the z-axis
        save_path (str): The path to save the plot if `plot` is True. Otherwise, it is not used
    """

    scale: torch.Tensor
    progress_bar: bool
    plot: bool
    train_rmse_x: list[float]
    train_rmse_y: list[float]
    train_rmse_z: list[float]
    validation_rmse_x: list[float]
    validation_rmse_y: list[float]
    validation_rmse_z: list[float]
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
        self.save_path = save_path

        self.train_rmse_x = []
        self.train_rmse_y = []
        self.train_rmse_z = []

        self.validation_rmse_x = []
        self.validation_rmse_y = []
        self.validation_rmse_z = []

    def on_train_epoch_end(
        self, trainer: Trainer, radlocatormodule: RadLocatorNetworkModule
    ) -> None:
        """Compute the scaled MSE at the end of the epoch for each axis

        Args:
            trainer (Trainer): The trainer
            radlocatormodule (RadLocatorNetworkModule): The model
        """
        output = torch.cat(radlocatormodule.train_step_outputs)
        target = torch.cat(radlocatormodule.train_step_targets)
        output = output * self.scale.to(output.device)
        target = target * self.scale.to(target.device)

        xaxis_error = torch.sqrt(((output[:, 0] - target[:, 0]) ** 2).mean())
        yaxis_error = torch.sqrt(((output[:, 1] - target[:, 1]) ** 2).mean())
        zaxis_error = torch.sqrt(((output[:, 2] - target[:, 2]) ** 2).mean())

        radlocatormodule.log_dict(
            {
                "train_scaled_rmse_x": xaxis_error,
                "train_scaled_rmse_y": yaxis_error,
                "train_scaled_rmse_z": zaxis_error,
            },
            on_epoch=True,
            prog_bar=self.progress_bar,
        )

        if self.plot and not trainer.sanity_checking:
            self.train_rmse_x.append(xaxis_error.item())
            self.train_rmse_y.append(yaxis_error.item())
            self.train_rmse_z.append(zaxis_error.item())

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

        xaxis_error = torch.sqrt(((output[:, 0] - target[:, 0]) ** 2).mean())
        yaxis_error = torch.sqrt(((output[:, 1] - target[:, 1]) ** 2).mean())
        zaxis_error = torch.sqrt(((output[:, 2] - target[:, 2]) ** 2).mean())

        radlocatormodule.log_dict(
            {
                "validation_scaled_rmse_x": xaxis_error,
                "validation_scaled_rmse_y": yaxis_error,
                "validation_scaled_rmse_z": zaxis_error,
            },
            on_epoch=True,
            prog_bar=self.progress_bar,
        )

        if self.plot and not trainer.sanity_checking:
            self.validation_rmse_x.append(xaxis_error.item())
            self.validation_rmse_y.append(yaxis_error.item())
            self.validation_rmse_z.append(zaxis_error.item())

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

        xaxis_error = torch.sqrt(((output[:, 0] - target[:, 0]) ** 2).mean())
        yaxis_error = torch.sqrt(((output[:, 1] - target[:, 1]) ** 2).mean())
        zaxis_error = torch.sqrt(((output[:, 2] - target[:, 2]) ** 2).mean())

        radlocatormodule.log_dict(
            {
                "test_scaled_rmse_x": xaxis_error,
                "test_scaled_rmse_y": yaxis_error,
                "test_scaled_rmse_z": zaxis_error,
            },
            on_epoch=True,
            prog_bar=self.progress_bar,
        )

    def on_fit_end(self, trainer: Trainer, pl_module: RadLocatorNetworkModule) -> None:
        """Plot the training and validation RMSE at the end of the training if `plot` is True

        Args:
            trainer (Trainer): The trainer (not used)
            pl_module (RadLocatorNetworkModule): The model (not used)
        """
        if self.plot:
            training_plot_axis(
                (self.train_rmse_x, self.train_rmse_y, self.train_rmse_z),
                (
                    self.validation_rmse_x,
                    self.validation_rmse_y,
                    self.validation_rmse_z,
                ),
                "Training and Validation RMSE per axis",
                "mm",
                self.save_path,
            )
