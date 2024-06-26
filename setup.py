from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("VERSION", "r", encoding="utf-8") as f:
    version = f.read()

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
    install_requires=[],
    classifiers=(
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ),
)
