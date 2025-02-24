# Installation

## Using ```pip```

```bash
pip install mcsm-benchs
```

## Installation using `poetry` for development mode

We use [```poetry```](https://python-poetry.org/docs/), a tool for dependency management and packaging in python to install the benchmarking framework. You can install ```poetry``` following the steps described [here](https://python-poetry.org/docs/#installation).
Then, make ```poetry``` create a virtual environment and install the main dependencies of the benchmarks using:

```bash
poetry install --with docs
```

If you have [`Anaconda`](https://www.anaconda.com/) or [`Miniconda`](https://docs.conda.io/en/latest/miniconda.html) installed please disable the auto-activation of the base environment and your conda environment using:

```bash
conda config --set auto_activate_base false
conda deactivate
```
