import os
import numpy as np
from torch.utils.data import Dataset
import h5py
import torch


class ConvDatasetRadLocatorNet(Dataset):
    """Dataset that uses the signals reshaped. Useful for convolutional networks

    Attributes:
        data_path (os.PathLike): The path to the data
        transforms (list[callable] | None): The list of transforms to apply to the data
        signals (np.ndarray): The signals
        labels (np.ndarray): The labels
    """

    data_path: os.PathLike
    transforms: torch.nn.Module | None
    signals: np.ndarray
    labels: np.ndarray

    def __init__(
        self,
        data_path: os.PathLike,
        transforms: torch.nn.Module | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize the dataset. The labels are always normalized to be between 0 and 1.

        Args:
            data_path (os.PathLike): The path to the data
            transforms (torch.nn.Module | None, optional): The Sequence of transforms to apply to the data. Defaults to None.
            dtype (np.dtype, optional): The data type of the data. Defaults to np.float32.
        """
        self.data_path = data_path
        self.transforms = transforms

        with h5py.File(self.data_path, "r") as f:
            self.signals = f["signals"][:].astype(dtype)
            step_size = f.attrs["step_size"]
            size = f.attrs["size"]
            self.labels = (f["labels"] * step_size / size)[:].astype(dtype)
            # Transform the signal and label to torch tensors
            self.signals = torch.from_numpy(self.signals)
            self.labels = torch.from_numpy(self.labels)

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

        if self.transforms:
            for transform in self.transforms:
                signal = transform(signal)

        return signal, label

    def __repr__(self) -> str:
        """Return the representation of the dataset.  Informs of the data path and the shape of the data, as well as the type of the data

        Returns:
            str: The representation of the dataset
        """
        return f"ConvDataset(data_path={self.data_path}, num_signals={self.signals.shape[2]}, num_samples={self.signals.shape[0]}, sample_length={self.signals.shape[1]}, num_labels={self.labels.shape[1]})"
