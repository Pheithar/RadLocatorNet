import torch


class RandomTurnOffTransform:
    """Transform that randomly turns off a signal with a given probability.

    Attributes:
        p (float): The probability of turning off the signal
    """

    def __init__(self, p: float) -> None:
        """Initialize the transform.

        Args:
            p (float): The probability of turning off the signal
        """
        self.p = p

    def __call__(self, signals: torch.Tensor) -> torch.Tensor:
        """Apply the transform. It is not applied to all the signals, just to some of them, randomly.
        The signals have a dimension (num_signals, length). The transform should turn off the signals with a probability `p`.

        Args:
            signals (torch.Tensor): The signals to apply the transform to

        Returns:
            torch.Tensor: The signals with the transform applied
        """
        mask = torch.rand(signals.shape[0]) < self.p
        signals[mask] = 0
        return signals
