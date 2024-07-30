import torch


class RandomNoiseTransform(torch.nn.Module):
    """Apply random noise to the signals.

    Attributes:
        noise (float): The noise to apply to the signals
    """

    def __init__(self, noise: float) -> None:
        """Initialize the transform.

        Args:
            noise (float): The noise to apply to the signals
        """
        super().__init__()
        self.noise = noise

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Apply the transform.

        Args:
            signals (torch.Tensor): The signals to apply the noise to

        Returns:
            torch.Tensor: The signals with noise applied
        """

        return signals + self.noise * torch.randn_like(signals)

    def __repr__(self):
        return f"RandomNoiseTransform(noise={self.noise})"
