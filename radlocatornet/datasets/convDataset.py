import os
from typing import Optional

import numpy as np
from torch.utils.data import Dataset


class ConvDataset(Dataset):
    """Dataset that uses the signals reshaped. Useful for convolutional networks

    Attributes:
        data_path (os.PathLike): The path to the data
        num_signals (int): The number of signals. Variable to reshape the data. Depends on the sensor
        transforms (list[callable] | None): The list of transforms to apply to the data
        data (np.ndarray): The data, loaded from the data path. It is expected to be 2 dimensional, with the first 3 columns being the labels and the rest being the flattened signals. It has to be able to be reshaped to the correct number of signals
        shape (tuple[int, int, int]): The shape of the data
    """

    def __init__(
        self,
        data_path: os.PathLike,
        num_signals: int,
        transforms: list[callable] | None = None,
        dtype: str = "float32",
    ) -> None:
        """Initialize the dataset

        Args:
            data_path (os.PathLike): The path to the data
            num_signals (int): The number of signals. If the data cannot be divided by the number of signals, it will throw an error
            transforms (list[callable] | None, optional): The list of transforms to apply to the data. Defaults to None.
            dtype (str, optional): The data type of the data. It has to be a valid numpy data type. Defaults to "float32".

        Raises:
            ValueError: If the data is not divisible by the number of signals
        """
        self.data_path = data_path
        self.num_signals = num_signals
        self.transforms = transforms

        self.data = np.load(self.data_path)

        # Test that the size make sense:
        if (self.data.shape[1] - 3) % num_signals != 0:
            raise ValueError(
                f"Data shape is not divisible by the number of signals. Expected {self.data.shape[1] - 3} % {num_signals} == 0"
            )

        self.shape = (
            self.data.shape[0],
            num_signals,
            (self.data.shape[1] - 3) // num_signals,
        )

        self.dtype = dtype

    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the item at the given index. The first 3 columns are the labels, the rest are the signals that are reshaped to the correct number of signals

        Args:
            idx (int): The index of the item

        Returns:
            tuple[np.ndarray, np.ndarray]: The signal and the label
        """
        label = self.data[idx, :3].astype(self.dtype)
        signal = self.data[idx, 3:].astype(self.dtype)

        # reshape signal to have the correct number of signals
        signal = signal.reshape(-1, self.num_signals).T

        return signal, label

    def __repr__(self) -> str:
        """Return the representation of the dataset.  Informs of the data path and the shape of the data, as well as the type of the data

        Returns:
            str: The representation of the dataset
        """
        return f"ConvDataset(data_path={self.data_path}, shape={self.shape}, dtype={self.dtype})"
