from typing import Dict

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from radlocatornet import utils
import torch


class Conv1DRadLocatorNet(nn.Module):
    """PyTorch model for the RadLocator project. It is a simple convolutional network that can be initialized with the number of signals, the layers, and the parameters.

    The structure is always the same:

    .. code-block::
        Conv1D -> BatchNorm1D -> Activation -> Dropout -> ... -> Flatten -> Linear -> Activation -> Linear -> Last Activation

    Attributes:
        model (nn.Sequential): The model
    """

    model: nn.Sequential

    def __init__(
        self,
        input_channels: int,
        conv_layers: list[dict[str, int | str]],
        fc_in_size: int,
        fc_layers: list[dict[str, int | str]],
        last_activation: str,
    ):
        """Initialize the model. The layers should be a list of dictionaries, where each dictionary has:

        For the convolutional layers, the dictionary should have the following keys:

            - channels: The number of channels
            - kernel_size: The size of the kernel
            - stride: The stride of the kernel
            - padding: The padding of the kernel
            - activation: The activation function
            - dropout: The dropout rate. If 0, no dropout is used
            - batch_norm: Whether to use batch normalization

        For the fully connected layers, the dictionary should have the following keys:

            - size: The size of the layer
            - activation: The activation function
            - dropout: The dropout rate. If 0, no dropout is used
            - batch_norm: Whether to use batch normalization

        Args:
            input_channels (int): The number of input channels
            conv_layers (list[dict[str, int | str]]): The convolutional layers
            fc_in_size (int): The size of the input of the fully connected layers. This is the output of the convolutional layers after flattening
            fc_layers (list[dict[str, int | str]]): The fully connected layers
            last_activation (str): The activation function of the last layer
        """
        super().__init__()
        self.model = nn.Sequential()

        # get_activation_function

        for i, layer in enumerate(conv_layers):
            self.model.add_module(
                f"conv_{i}",
                nn.Conv1d(
                    input_channels,
                    layer["channels"],
                    layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                ),
            )

            if layer.get("batch_norm", False):
                self.model.add_module(
                    f"conv_batch_norm_{i}", nn.BatchNorm1d(layer["channels"])
                )

            self.model.add_module(
                f"conv_activation_{i}",
                utils.get_activation_function(layer["activation"]),
            )

            if layer.get("dropout", 0) > 0:
                self.model.add_module(f"conv_dropout_{i}", nn.Dropout(layer["dropout"]))

            input_channels = layer["channels"]

        self.model.add_module("flatten", nn.Flatten())

        # The input size of the fully connected layers is the output size of the convolutional layers
        input_channels = fc_in_size

        for i, layer in enumerate(fc_layers):
            self.model.add_module(f"fc_{i}", nn.Linear(input_channels, layer["size"]))

            if layer.get("batch_norm", False):
                self.model.add_module(
                    f"fc_batch_norm_{i}", nn.BatchNorm1d(layer["size"])
                )

            self.model.add_module(
                f"fc_activation_{i}", utils.get_activation_function(layer["activation"])
            )
            if layer.get("dropout", 0) > 0:
                self.model.add_module(f"fc_dropout_{i}", nn.Dropout(layer["dropout"]))

            input_channels = layer["size"]

        self.model.add_module(
            "output",
            nn.Linear(input_channels, 3),
        )

        self.model.add_module(
            "last_activation",
            utils.get_activation_function(last_activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        return self.model(x)

    def __repr__(self) -> str:
        """Representation of the model. We just used the default representation of the Sequential model

        Returns:
            str: The representation of the model
        """

        return self.model.__repr__()
