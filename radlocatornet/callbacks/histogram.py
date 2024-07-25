from lightning import Callback, Trainer
from radlocatornet.models import RadLocatorNetworkModule
from irase_plotting import sensor
import os
import torch
import numpy as np


class HistogramCallback(Callback):
    """Callback to plot a histogram of the prediction positions during training. The callback is called at the end of the epoch, and at the end of the validation, train and test steps, and plots the histogram.

    These callbacks are to be used with the RadLocatorNet Model Module, as they need `train_step_outputs`, `evaluation_step_outputs` and `test_step_outputs` to be present in the model, as well as `train_step_targets`, `evaluation_step_targets` and `test_step_targets`.

    Attributes:
        save_path (str): The path to save the plot
        detector_size (tuple[float, float, float]): The size of the detector sensor
        num_anodes (int): The number of anodes in the sensor
        num_cathodes (int): The number of cathodes in the sensor
        num_drifts (int): The number of drifts in the sensor
        bins (tuple[int, int, int]): The number of bins for each axis
    """

    save_path: str
    detector_size: np.ndarray
    num_anodes: int
    num_cathodes: int
    num_drifts: int
    bins: tuple[int, int, int]

    def __init__(
        self,
        save_path: str,
        detector_size: tuple[float, float, float],
        num_anodes: int,
        num_cathodes: int,
        num_drifts: int,
        bins: tuple[int, int, int],
    ) -> None:
        """Initialize the callback

        Args:
            save_path (str): The path to save the plot. Must be a directory, not a file
            detector_size (tuple[float, float, float]): The size of the detector sensor
            num_anodes (int): The number of anodes in the sensor
            num_cathodes (int): The number of cathodes in the sensor
            num_drifts (int): The number of drifts in the sensor
            bins (tuple[int, int, int]): The number of bins for each axis

        Raises:
            ValueError: If the `save_path` is not a directory
        """

        if not os.path.isdir(save_path):
            raise ValueError("The `save_path` must be a directory, not a file")

        self.save_path = save_path
        self.detector_size = np.array(detector_size)
        self.num_anodes = num_anodes
        self.num_cathodes = num_cathodes
        self.num_drifts = num_drifts
        self.bins = bins

    def on_train_epoch_end(
        self, trainer: Trainer, radlocatormodule: RadLocatorNetworkModule
    ) -> None:
        """Plot the histogram of the prediction positions during training

        Args:
            trainer (Trainer): The trainer
            radlocatormodule (RadLocatorNetworkModule): The model module
        """

        epoch = trainer.current_epoch
        file_name = f"train_epoch_{epoch}_histogram.png"

        output = torch.cat(radlocatormodule.train_step_outputs)
        output = output.detach().cpu().numpy()
        output = output * self.detector_size

        sensor.histogram(
            output,
            detector_size=self.detector_size,
            num_anodes=self.num_anodes,
            num_cathodes=self.num_cathodes,
            num_drifts=self.num_drifts,
            bins=self.bins,
            show=False,
            save_path=os.path.join(self.save_path, file_name),
            view="all",
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, radlocatormodule: RadLocatorNetworkModule
    ) -> None:
        """Plot the histogram of the prediction positions during validation

        Args:
            trainer (Trainer): The trainer
            radlocatormodule (RadLocatorNetworkModule): The model module
        """

        epoch = trainer.current_epoch
        file_name = f"validation_epoch_{epoch}_histogram.png"

        output = torch.cat(radlocatormodule.evaluation_step_outputs)
        output = output.detach().cpu().numpy()
        output = output * self.detector_size

        sensor.histogram(
            output,
            detector_size=self.detector_size,
            num_anodes=self.num_anodes,
            num_cathodes=self.num_cathodes,
            num_drifts=self.num_drifts,
            bins=self.bins,
            show=False,
            save_path=os.path.join(self.save_path, file_name),
            view="all",
        )

    def on_test_epoch_end(
        self, trainer: Trainer, radlocatormodule: RadLocatorNetworkModule
    ) -> None:
        """Plot the histogram of the prediction positions during testing

        Args:
            trainer (Trainer): The trainer
            radlocatormodule (RadLocatorNetworkModule): The model module
        """

        file_name = "test_histogram.png"

        output = torch.cat(radlocatormodule.test_step_outputs)
        output = output.detach().cpu().numpy()
        output = output * self.detector_size

        sensor.histogram(
            output,
            detector_size=self.detector_size,
            num_anodes=self.num_anodes,
            num_cathodes=self.num_cathodes,
            num_drifts=self.num_drifts,
            bins=self.bins,
            show=False,
            save_path=os.path.join(self.save_path, file_name),
            view="all",
        )
