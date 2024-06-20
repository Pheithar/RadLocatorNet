"""File to control additional or secondary functions needed for the models"""

from torch import nn


def get_activation_function(name: str) -> nn.Module:
    """Get the activation function from the name

    Allowed activation functions:
        - relu -> nn.ReLU
        - tanh -> nn.Tanh
        - sigmoid -> nn.Sigmoid
        - identity -> nn.Identity


    Args:
        name (str): The name of the activation function

    Returns:
        nn.Module: The activation function

    Raises:
        ValueError: If the activation function is not found
    """
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity(),
    }
    if name in activations:
        return activations[name]

    raise ValueError(f"Activation function '{name}' not found or not yet implemented")


def get_loss_function(name: str) -> nn.Module:
    """Get the loss function from the name

    Allowed loss functions:
        - mse -> nn.MSELoss
        - l1 -> nn.L1Loss

    Args:
        name (str): The name of the loss function

    Returns:
        nn.Module: The loss function

    Raises:
        ValueError: If the loss function is not found
    """
    losses = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
    }
    if name in losses:
        return losses[name]

    raise ValueError(f"Loss function '{name}' not found or not yet implemented")
