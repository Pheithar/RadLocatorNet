import torch


class RandomNoiseTransform:
    """Apply random noise to the signals.

    Attributes:
        noise (float): The noise to apply to the signals
    """

    def __init__(self, noise: float) -> None:
        """Initialize the transform.

        Args:
            noise (float): The noise to apply to the signals
        """
        self.noise = noise

    def __call__(self, signals: torch.Tensor) -> torch.Tensor:
        """Apply the transform.

        Args:
            signals (torch.Tensor): The signals to apply the noise to

        Returns:
            torch.Tensor: The signals with noise applied
        """

        return signals + self.noise * torch.randn_like(signals)
