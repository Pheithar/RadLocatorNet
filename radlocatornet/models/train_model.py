"""
File to train the different models. The models have to be loaded from the config file using **hydra**.
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    """Test the configuration file.

    Args:
        cfg (DictConfig): Configuration file.
    """
    print(OmegaConf.to_yaml(cfg))


def test_function(a: int, b: int) -> float:
    """Test function. I just want to see some text.
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam nec

    Args:
        a (int): First number. Cool
        b (int): Second number.

    Returns:
    --------
    float: Multiplication of the two numbers.

    """
    print("Test function")
    return a * b


def test2_function(a: int, b: int, c: int) -> float:
    """Test function. I just want to see some text.
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam nec

    Args:
        a (int): First number.
        b (int): Second number.
        c (int): Third number.

    Returns:
    --------
    float: Multiplication of the two numbers.

    """
    print("Test function")
    return a * b


def test3(a: float) -> str:
    """_summary_

    Args:
        a (float): _description_

    Returns:
        str: _description_
    """
    return "a"


if __name__ == "__main__":
    my_app()
