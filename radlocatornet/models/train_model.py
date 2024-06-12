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


def test_function():
    """Test function."""
    print("Test function")


if __name__ == "__main__":
    my_app()
