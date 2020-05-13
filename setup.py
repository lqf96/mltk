#! /usr/bin/env python3
from setuptools import setup

setup(
    name="mltk",
    version="0.1.0",
    description="Machine learning utilities tookit built on PyTorch",
    author="Qifan Lu",
    author_email="lqf.1996121@gmail.com",
    url="https://github.com/lqf96/mltk",
    packages=["mltk"],
    install_requires=[
        "numpy",
        "sortedcontainers",
        "tensorboard",
        "torch",
        "typing-extensions"
    ]
)
