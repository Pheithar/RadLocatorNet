import os
import numpy as np
from torch.utils.data import Dataset
import h5py


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
        transforms: list[callable] | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize the dataset

        Args:
            data_path (os.PathLike): The path to the data
            transforms (list[callable] | None, optional): The list of transforms to apply to the data. Defaults to None.
            dtype (np.dtype, optional): The data type of the data. Defaults to np.float32.
        """
        self.data_path = data_path
        self.transforms = transforms

        with h5py.File(self.data_path, "r") as f:
            self.signals = f["signals"][:].astype(dtype)
            step_size = f.attrs["step_size"]
            size = f.attrs["size"]
            self.labels = (f["labels"] * step_size / size)[:].astype(dtype)

    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the item at the given index. The signal have to be transposed, as they come in (num_samples, sample_length, num_signals), but PyTorch expects (num_signals, sample_length)

        Args:
            idx (int): The index of the item

        Returns:
            tuple[np.ndarray, np.ndarray]: The signal and the label
        """
        signal = self.signals[idx].T
        label = self.labels[idx]

        return signal, label

    def __repr__(self) -> str:
        """Return the representation of the dataset.  Informs of the data path and the shape of the data, as well as the type of the data

        Returns:
            str: The representation of the dataset
        """
        return f"ConvDataset(data_path={self.data_path}, shape={self.shape})"
