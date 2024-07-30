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
    if name.lower() in activations:
        return activations[name.lower()]

    raise ValueError(f"Activation function '{name}' not found or not yet implemented")
