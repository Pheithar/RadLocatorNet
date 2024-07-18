import os
from typing import Optional

import numpy as np
from torch.utils.data import Dataset


class FlatDataset(Dataset):
    """Dataset that uses the signals flattened. Useful for fully connected networks

    Attributes:
        data_path (os.PathLike): The path to the data
        transforms (list[callable] | None) A list of transforms to apply to the data. Defaults to None.
        data (np.ndarray): The data, loaded from the data path. It is expected to be 2 dimensional, with the first 3 columns being the labels and the rest being the flattened signals.
        shape (tuple[int, int]): The shape of the data
        dtype (str): The data type of the data. It has to be a valid numpy data type
    """

    data_path: os.PathLike
    transforms: list[callable] | None
    data: np.ndarray
    shape: tuple[int, int]
    dtype: str

    def __init__(
        self,
        data_path: os.PathLike,
        transforms: list[callable] | None = None,
        dtype: str = "float32",
    ) -> None:
        """Initialize the dataset

        Args:
            data_path (os.PathLike): The path to the data. Expects a numpy file, or something that can be loaded with np.load
            transforms (list[callable] | None, optional): A list of transforms to apply to the data. Defaults to None.
            dtype (str, optional): The data type of the data. It has to be a valid numpy data type. Defaults to "float32".

        Raises:
            ValueError: If the data is not 2 dimensional
        """
        self.data_path = data_path
        self.transforms = transforms

        self.data = np.load(self.data_path)
        self.shape = self.data.shape

        if len(self.shape) != 2:
            raise ValueError(
                f"Data shape is expected to be 2 dimensional, but got {len(self.shape)}"
            )

        self.dtype = dtype

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: The length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get the label and the flattened signals from a given index. The first 3 columns are the labels, the rest are the signals, that are flattened.

        Args:
            idx (int): The index of the data

        Returns:
            tuple[np.ndarray, np.ndarray]: The signal and the label
        """
        label = self.data[idx, :3].astype(self.dtype)
        signal = self.data[idx, 3:].astype(self.dtype)

        print(f"Signal: {signal.shape}, Label: {label.shape}")

        print(f"Signal: {signal.min()}, {signal.max()}")

        assert False
        return signal, label

    def __repr__(self) -> str:
        """Return the representation of the dataset. Informs of the data path and the shape of the data, as well as the type of the data

        Returns:
            str: The representation of the dataset
        """
        return f"FlatDataset(data_path={self.data_path}, shape={self.shape}, dtype={self.dtype})"
