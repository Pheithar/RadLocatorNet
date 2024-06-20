import lightning as L
from radlocatornet.datasets import ConvDataset, FlatDataset
import torch
from torch.utils.data import random_split, DataLoader


class RadLocatorDataModule(L.LightningDataModule):
    """Data module for the RadLocator project. This class should be used to load the data and prepare it for the model. The data module should be used to load the data, split it into training, validation, and test sets, and prepare it for the model.

    Attributes:
        type (str): The type of the dataset
        data_path (str): Location of the data
        batch_size (int): The batch size
        num_workers (int): Number of worker for the data loader
        num_signals (int, optional): Number of signals. Necessary for the `conv` type. Defaults to -1.
        transforms (list[callable] | None): List of transforms to apply to the data. Defaults to None.
        dtype (str): Precision of the data. Defaults to "float32"
        dataset (torch.utils.data.Dataset): The dataset
        generator (torch.Generator): The random number generator
        data_split (list[float]): The split of the data. Should be a list of 3 floats, representing the training, validation, and test split. It should add up to 1
        train_set (torch.utils.data.Subset): The training set
        val_set (torch.utils.data.Subset): The validation set
        test_set (torch.utils.data.Subset): The test set
    """

    def __init__(
        self,
        type: str,
        data_path: str,
        batch_size: int,
        num_workers: int,
        data_split: list[float],
        num_signals: int = -1,
        transforms: list[callable] | None = None,
        dtype: str = "float32",
        seed: int = 42,
    ) -> None:
        """Initialize the data module. If the type is `conv`, there should be a number of signals. If the number of signals is not provided, it will raise an error.

        Allowed types are:
            - flat -> FlatDataset
            - conv -> ConvDataset

        Args:
            type (str): The type of the dataset
            data_path (str): Location of the data
            batch_size (int): The batch size
            num_workers (int): Number of worker for the data loader
            data_split (list[float]): The split of the data. Should be a list of 3 floats, representing the training, validation, and test split, in that order. It should add up to 1
            num_signals (int, optional): Number of signals. Necessary for the `conv` type. Defaults to -1.
            transforms (list[callable] | None, optional): List of transforms to apply to the data. Defaults to None.
            dtype (str, optional): Precision of the data. Defaults to "float32".
            seed (int, optional): Seed for the random number generator. Defaults to 42.

        Raises:
            ValueError: If the data split does not add up to 1
        """
        super().__init__()

        self.type = type
        self.data_path = data_path
        self.transforms = transforms
        self.dtype = dtype
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_signals = num_signals
        self.generator = torch.Generator().manual_seed(seed)

        if sum(data_split) != 1:
            raise ValueError("Data split should add up to 1")

        self.data_split = data_split

    def prepare_data(self) -> None:
        """Prepare the data. This method should be used to load the data and prepare it for the model. The data should be loaded here, but not split into training, validation, and test sets. This should be done in the `setup` method.

        Raises:
            ValueError: If the type of the dataset is not recognized
            ValueError: If the number of signals is not provided for the ConvDataset
        """
        if self.type == "flat":
            self.dataset = FlatDataset(self.data_path, self.transforms, self.dtype)
        elif self.type == "conv":
            if self.num_signals == -1:
                raise ValueError("Number of signals is required for the ConvDataset")
            self.dataset = ConvDataset(
                self.data_path, self.num_signals, self.transforms, self.dtype
            )
        else:
            raise ValueError(f"Dataset type '{type}' not recognized")

    def setup(self, stage=None) -> None:
        """Setup the data. This method should be used to split the data into training, validation, and test sets. The data should be split here, and the splits should be saved as attributes of the class. The splits should be accessible through the `train_dataloader`, `val_dataloader`, and `test_dataloader` methods.

        Args:
            stage ([type], optional): The stage of the setup. Defaults to None.
        """
        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset, self.data_split, generator=self.generator
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training data loader

        Returns:
            DataLoader: The training data loader, with the training set. It has shuffling enabled
        """
        print(self.train_set)
        print("_________________________________")
        print(self.batch_size)
        print("_________________________________")
        print(self.num_workers)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation data loader

        Returns:
            DataLoader: The validation data loader, with the validation set. It has shuffling disabled
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test data loader

        Returns:
            DataLoader: The test data loader, with the test set. It has shuffling disabled
        """
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
