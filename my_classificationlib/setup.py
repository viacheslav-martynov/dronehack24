from typing import List

from setuptools import find_packages, setup

from my_classificationlib import __version__


def get_requirements(path: str) -> List[str]:
    with open(path, "r") as file:
        requirements = file.read().split()

    return requirements


__version__ = "0.0.1"

setup(
    name="my_classificationlib",
    version=__version__,
    description="",
    author="",
    packages=find_packages(),
    python_requires=">=3.7",
    keywords=["pytorch", "classification"],
    install_requires=get_requirements("requirements/main.txt"),
    extras_require={
        "optional": get_requirements("requirements/optional.txt"),
    },
)
