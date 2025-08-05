"""
Setup configuration for the NFL Predictor package.

This module configures the package installation, including dependencies,
metadata, and Python version requirements.
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("VERSION", "r", encoding="utf-8") as f:
    version = f.read().strip()

setup(
    name="nfl-predictor",
    version=version,
    author="Mitch Avis",
    author_email="mitchavis@gmail.com",
    description="Eventually, something useful for predicting football games",
    long_description=long_description,
    license="MIT",
    url="https://github.com/mitch-avis/nfl-predictor",
    packages=find_packages(),
    python_requires=">=3.12",
    keywords="stats sports football machine learning",
    install_requires=[
        "beautifulsoup4",
        "coloredlogs",
        "lxml",
        "numpy",
        "pandas",
        "pyarrow",
        "requests",
        "scikit-learn",
        "scipy",
        "xgboost",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
