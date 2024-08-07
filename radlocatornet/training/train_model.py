"""File to kick off the training of a model from a single place"""

# OLD above
import lightning as L
from torch import nn
from radlocatornet import models
from radlocatornet.datasets.dataModule import RadLocatorDataModule
from pytorch_lightning.loggers import Logger, CSVLogger


def train_model(
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: nn.Module,
    trainer_cfg: dict[str, any],
    dataloader_cfg: dict[str, any],
    dataset: nn.Module,
    logger: Logger | None = None,
    callbacks: list[L.Callback] | None = None,
) -> None:
    """Train the model in the iRASE setting

    Args:
        model (nn.Module): PyTorch model to train
        loss_function (nn.Module): The loss function
        optimizer (nn.Module): The optimizer
        trainer_cfg (dict[str, any]): The configuration of the trainer. Must be in accordance with PyTorch Lightning Trainer (https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)
        dataloader_cfg (dict[str, any]): The configuration of the dataloader. Must be in accordance with RadLocatorDataModule (radlocatornet.datasets.dataModule.RadLocatorDataModule)
        dataset (nn.Module): The dataset to train on
        logger (Logger | None, optional): The logger to use during training. Defaults to None. Must be in accordance with PyTorch Lightning Logger (https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html)
        callbacks (list[L.Callback] | None, optional): The callbacks to use during training. Defaults to None.
    """

    model_module = models.RadLocatorNetworkModule(model, loss_function, optimizer)

    trainer = L.Trainer(
        **trainer_cfg,
        logger=logger,
        callbacks=callbacks,
    )

    datamodule = RadLocatorDataModule(dataset, **dataloader_cfg)

    trainer.fit(model_module, datamodule=datamodule)
    trainer.test(model_module, datamodule=datamodule)
