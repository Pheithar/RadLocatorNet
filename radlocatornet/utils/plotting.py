"""Function used for plotting the results of the training"""

import matplotlib.pyplot as plt
import os


def training_plotting(
    train_values: list[float],
    validation_values: list[float],
    title: str,
    ylabel: str,
    save_path: os.PathLike,
) -> None:
    """Plot the training and validation values. The idea of this function is to maintain consistent plots for different parts of the code.

    ..warning::
        This function can only plot one value for the training and validation. If more values are needed, use the `training_plot_axis` function.

    Args:
        train_values (list[float]): The training values
        validation_values (list[float]): The validation values
        title (str): The title of the plot
        ylabel (str): The ylabel of the plot
        save_path (os.PathLike): The path to save the plot
    """

    plt.figure(figsize=(10, 6))

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.plot(train_values, label="Train", color="teal")
    plt.plot(validation_values, label="Validation", color="orange")
    plt.legend()

    # Check if path is dir or file
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, f"{title.replace(' ', '_')}.png")

    plt.savefig(save_path, bbox_inches="tight", dpi=500)
    plt.close()


def training_plot_axis(
    train_values: tuple[list[float], list[float], list[float]],
    validation_values: tuple[list[float], list[float], list[float]],
    title: str,
    ylabel: str,
    save_path: os.PathLike,
):
    """Plot the training and validation values. The idea of this function is to maintain consistent plots for different parts of the code. This function expects the values for each of the axis, ordered as x, y and z. It uses the same color for validation and train, but different line styles.

    Args:
        train_values (tuple[list[float], list[float], list[float]]): The training values for each axis. The first element is the x-axis, the second is the y-axis, and the third is the z-axis
        validation_values (tuple[list[float], list[float], list[float]]): The validation values for each axis. The first element is the x-axis, the second is the y-axis, and the third is the z-axis
        title (str): The title of the plot
        ylabel (str): The unit of the values
        save_path (os.PathLike): The path to save the plot
    """
    plt.figure(figsize=(10, 6))

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.plot(train_values[0], label="Train X", linestyle="-", color="crimson")
    plt.plot(
        validation_values[0], label="Validation X", linestyle="--", color="crimson"
    )
    plt.plot(train_values[1], label="Train Y", linestyle="-", color="#DAA520")
    plt.plot(
        validation_values[1], label="Validation Y", linestyle="--", color="#DAA520"
    )
    plt.plot(train_values[2], label="Train Z", linestyle="-", color="#9370DB")
    plt.plot(
        validation_values[2], label="Validation Z", linestyle="--", color="#9370DB"
    )

    plt.legend()

    # Check if path is dir or file
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, f"{title.replace(' ', '_')}.png")

    plt.savefig(save_path, bbox_inches="tight", dpi=500)
    plt.close()
