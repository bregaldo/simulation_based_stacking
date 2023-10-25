# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sbi_stacking",
    version="1.0",
    author="Anonymous",
    author_email="bregaldo@flatironinstitute.org",
    description="Stacking for SBI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bregaldo/simulation_based_stacking",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
