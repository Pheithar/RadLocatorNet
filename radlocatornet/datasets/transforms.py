""" All the custom available transforms for the dataset """

import torch
from torch import nn


class NormalizeLabel(nn.Module):
    """Normalize the label to be between 0 and 1 given an original range.

    Attributes:
        original_range (torch.Tensor): The original range of the labels.
    """

    def __init__(self, original_range: list[float]) -> None:
        """Initialize the transform. The expected input is a list of 3 floats, representing the maximum original value of the x, y, and z coordinates.

        Args:
            original_range (list[float]): The original range of the labels. Expected to be a list of 3 floats.

        Raises:
            ValueError: If the original range is not a list of 3 floats
        """
        super().__init__()

        if len(original_range) != 3:
            raise ValueError(
                f"Original range should be a list of 3 floats, but got {len(original_range)}"
            )

        self.original_range = torch.tensor(original_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the label to be between 0 and 1 given an original range.

        Args:
            x (torch.Tensor): The label to be normalized

        Returns:
            torch.Tensor: The normalized label
        """
        return x / self.original_range

    def __repr__(self) -> str:
        return f"NormalizeLabel (original_range={self.original_range})"


class RandomNoise(nn.Module):
    """Add random noise to the signal.

    Attributes:
        noise_level (float): The level of noise to add
    """

    def __init__(self, noise_level: float) -> None:
        """Initialize the transform.

        Args:
            noise_level (float): The level of noise to add, in percentage between 0 and 1
        """
        super().__init__()

        self.noise_level = noise_level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add random noise to the signal.

        Args:
            x (torch.Tensor): The signal

        Returns:
            torch.Tensor: The signal with added noise
        """
        return x + torch.randn_like(x) * self.noise_level

    def __repr__(self) -> str:
        return f"RandomNoise (noise_level={self.noise_level})"
