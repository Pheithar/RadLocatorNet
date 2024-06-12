from setuptools import setup, find_packages


VERSION = "0.0.1"
NAME = "radlocatornet"
DESCRIPTION = "Location of radiation source using neural network"
LONG_DESCRIPTION = "Location of radiation source using neural network"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Alejandro Valverde Mahou",
    packages=find_packages(),
    install_requires=[],  # TODO: Add dependencies
)
