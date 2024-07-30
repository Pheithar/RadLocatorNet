import lightning as L
import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset
import math


class RadLocatorDataModule(L.LightningDataModule):
    """Data module for the RadLocator project. This class should be used to load the data and prepare it for the model. The data module should be used to load the data, split it into training, validation, and test sets, and prepare it for the model.

    Attributes:
        dataset (Dataset): The dataset that should be used in the data module
        batch_size (int): The batch size
        num_workers (int): Number of workers for the data loader
        generator (torch.Generator): The random number generator
        data_split (list[float]): The split of the data. Should be a list of 3 floats, representing the training, validation, and test split. It should add up to 1
        train_set (Dataset): The training set
        val_set (Dataset): The validation set
        test_set (Dataset): The test set

    """

    dataset: Dataset
    batch_size: int
    num_workers: int
    generator: torch.Generator
    data_split: tuple[float, float, float]
    train_set: Dataset
    val_set: Dataset
    test_set: Dataset

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 8,
        data_split: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ) -> None:
        """Initialize the data module.

        Args:
            dataset (Dataset): The dataset that should be used in the data module
            batch_size (int): The batch size
            num_workers (int): Number of workers for the data loader. Defaults to 8.
            data_split (tuple[float, float, float]): The split of the data. Should be a list of 3 floats, representing the training, validation, and test split. It should add up to 1. Defaults to (0.8, 0.1, 0.1).
            seed (int, optional): The seed for the random number generator. Defaults to 42.

        Raises:
            ValueError: If the data split does not add up to 1
        """
        super().__init__()

        self.dataset = dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generator = torch.Generator().manual_seed(seed)

        if not math.isclose(sum(data_split), 1):
            raise ValueError(f"Data split should add up to 1. Got {sum(data_split)}")

        self.data_split = data_split

        # Save hyperparameters
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        """Prepare the data. This method should be used to load the data and prepare it for the model.

        .. warning::
            Right now this method does nothing. It is here for future use
        """
        pass

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

        .. note::
            The training data loader has shuffling and drop_last enabled
            This is to ensure that batch normalization works correctly
            and that the training set is not biased towards the last batch

        Returns:
            DataLoader: The training data loader, with the training set. It has shuffling enabled
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
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
