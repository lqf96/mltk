#! /usr/bin/env python3
from setuptools import setup

setup(
    name="mltk",
    version="0.1.0",
    description="Machine learning utilities tookit, built on PyTorch and Ignite",
    author="Qifan Lu",
    author_email="lqf96@uw.edu",
    url="https://github.com/lqf96/mltk",
    packages=["mltk"],
    install_requires=[
        "gym",
        "numpy",
        "pytorch-ignite",
        "tensorboard",
        "torch",
    ]
)
