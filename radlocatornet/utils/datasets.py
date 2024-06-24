""" Apply the transformations to the dataset, or raises an error if they are not valid """

import torch
from radlocatornet.datasets import NormalizeLabel


def apply_transforms(
    transforms: list[str], signal: torch.Tensor, label: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the transformations to the dataset, or raises an error if they are not valid.

    There are some transforms that have to be applied to bath the signal and the label, while some are only applied to the signal, and some only to the label. This function will apply all the transformations to the correct data.

    Args:
        transforms (list[str]): List of transformations to apply
        signal (torch.Tensor): The signal
        label (torch.Tensor): The label

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The transformed signal and label

    Raises:
        ValueError: If the transform is not a valid callable
    """
    for transform in transforms:
        if transform["name"] == "normalize_label":
            label = NormalizeLabel(transform["params"])(label)
        if 

    return signal, label
