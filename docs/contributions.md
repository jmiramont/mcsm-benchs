# How to contribute to `mscm-benchs`?

We welcome contributions that improve `mcsm-benchs`. From basic functionality to new examples, you can make your contributions following the guidelines below.

## Generalities

New functionality must be documented and covered by associated unit tests.

Tests should be run locally before making a pull-request.

Please check our [Code of Conduct](https://jmiramont.github.io/mcsm-benchs/code_of_conduct.html) before making contributions.

## Installation in development mode

Follow the [installation instructions](https://jmiramont.github.io/mcsm-benchs/install.html) using `poetry` for development mode.

## Documentation

All new functionality must be documented. Ideally, even for private functions, a docstring should describe how the function works and include a brief explanation of the input and output arguments.

You can locally build the documentation using [Sphinx](https://www.sphinx-doc.org/en/master/):

```bash
cp notebooks/ docs -r
cd docs
poetry run sphinx-apidoc -o . ../mcsm_benchs 
poetry run make clean
poetry run make html
```

## Running tests

Please include unit tests covering any additional functionality. Once the new tests are added to the `tests` folder, run `pytest` using

```bash
poetry run pytest
```

All tests must pass locally *before* making a pull-request.
