"""File to kick off the training of a model from a single place"""

from typing import Any
import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from radlocatornet.models import FullyConnectedNetwork, Conv1DNetwork
from radlocatornet.datasets import RadLocatorDataModule
from radlocatornet.utils import get_callbacks
from torch import nn


def train_model(
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    logger_cfg: dict[str, Any] | None = None,
    callbacks_cfg: list[dict[str, Any]] | None = None,
) -> nn.Module:
    """Train the model using the specified configuration. The training uses PyTorch Lightning, so the configuration should be in accordance with the PyTorch Lightning Trainer. The model should be in accordance with the models in `radlocatornet/models/`.

    Args:
        model_cfg (dict[str, Any]): Configureation of the model. Should be in accordance with the model type. See `radlocatornet/models/__init__.py` for available models, and `radlocatornet/models/` for the configuration of each model.
        training_cfg (dict[str, Any]): Configuration of the training. Should be in accordance with the PyTorch Lightning Trainer. See https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html for more information.
        data_cfg (dict[str, Any]): TODO: _description_
        logger_cfg (dict[str, Any] | None, optional): Configuration of the logger. Should be in accordance with the PyTorch Lightning Logger. See https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html. The available loggers are `wandb`, `tensorboard`, and `csv`. Defaults to None.
        callback_cfg (list[dict[str, Any]] | None, optional): Configuration of the callbacks. Should be in accordance with the PyTorch Lightning Callbacks. See https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html. Defaults to None.

    Returns:
        nn.Module: The trained model

    Raises:
        NotImplementedError: If the model type is not implemented. This is to ensure that the user knows that the training is not yet available.
    """
    print("Training the model")
    print(f"Model config: {model_cfg}")
    print(f"Training config: {training_cfg}")
    print(f"Data config: {data_cfg}")
    print(f"Logger config: {logger_cfg}")
    print(f"Callback config: {callbacks_cfg}")

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
    trainer.validate(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    return model
