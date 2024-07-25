"""File to kick off the training of a model from a single place"""

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger, CSVLogger, Logger
from radlocatornet.models import FullyConnectedNetwork, Conv1DNetwork
from radlocatornet.datasets import RadLocatorDataModule
from radlocatornet.utils import get_callbacks


# OLD above
import lightning as L
from torch import nn
from radlocatornet import models


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


def train_model_old(
    model_cfg: dict[str, any],
    training_cfg: dict[str, any],
    data_cfg: dict[str, any],
    logger_cfg: dict[str, any] | None = None,
    callbacks_cfg: list[dict[str, any]] | None = None,
) -> nn.Module:
    """Train the model using the specified configuration. The training uses PyTorch Lightning, so the configuration should be in accordance with the PyTorch Lightning Trainer. The model should be in accordance with the models in `radlocatornet/models/`.

    Args:
        model_cfg (dict[str, any]): Configuration of the model. Should be in accordance with the model type. See `radlocatornet/models/__init__.py` for available models, and `radlocatornet/models/` for the configuration of each model.
        training_cfg (dict[str, any]): Configuration of the training. Should be in accordance with the PyTorch Lightning Trainer. See https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html for more information.
        data_cfg (dict[str, any]): Configuration of the data.
        logger_cfg (dict[str, any] | None, optional): Configuration of the logger. Should be in accordance with the PyTorch Lightning Logger. See https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html. The available loggers are `wandb`, `tensorboard`, and `csv`. Defaults to None.
        callback_cfg (list[dict[str, any]] | None, optional): Configuration of the callbacks. Should be in accordance with the PyTorch Lightning Callbacks. See https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html. Defaults to None.

    Returns:
        nn.Module: The trained model

    Raises:
        NotImplementedError: If the model type is not implemented. This is to ensure that the user knows that the training is not yet available.
    """
    logger = None
    loggers = {
        "wandb": WandbLogger,
        "tensorboard": TensorBoardLogger,
        "csv": CSVLogger,
    }
    if logger_cfg:
        logger_type = logger_cfg.pop("type")

        if logger_type not in loggers:
            raise NotImplementedError(f"Logger type '{logger_type}' not implemented")

        logger = loggers[logger_type](**logger_cfg)

    callbacks = None
    if callbacks_cfg:
        callbacks = get_callbacks(callbacks_cfg)

    trainer = L.Trainer(
        **training_cfg,
        logger=logger,
        callbacks=callbacks,
    )

    model_type = model_cfg.pop("type")

    models = {
        "FCN": FullyConnectedNetwork,
        "CNN1D": Conv1DNetwork,
    }

    if model_type not in models:
        raise NotImplementedError(f"Model type '{model_type}' not implemented")

    model = models[model_type](**model_cfg)

    datamodule = RadLocatorDataModule(**data_cfg)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)
    return model
