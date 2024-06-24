from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint


def get_callbacks(callbacks: list[dict[str, any]]) -> list[Callback]:
    """Get the callbacks from the configuration

    Args:
        callbacks (list[dict[str, Any]]): The configuration of the callbacks

    Returns:
        list[L.Callback]: The callbacks
    """
    callback_classes = {
        "early_stopping": EarlyStopping,
        "model_checkpoint": ModelCheckpoint,
    }

    callback_list = []
    for callback in callbacks:
        callback_type = callback.pop("type")

        if callback_type not in callback_classes:
            raise NotImplementedError(
                f"Callback type '{callback_type}' not implemented"
            )

        callback_list.append(callback_classes[callback_type](**callback))

    return callback_list
