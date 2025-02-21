[![Tests](https://github.com/jmiramont/mcsm-benchs/actions/workflows/tests.yml/badge.svg)](https://github.com/jmiramont/mcsm-benchs/actions/workflows/tests.yml) [![codecov](https://codecov.io/gh/jmiramont/mcsm-benchs/graph/badge.svg?token=CJPPKYJD8H)](https://codecov.io/gh/jmiramont/mcsm-benchs) [![Documentation](docs/readme_figures/docs_badge.svg)](https://jmiramont.github.io/mcsm-benchs)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# `mcsm-benchs`: A Toolbox for Benchmarking Multi-Component Signal Analysis Methods

A public, open-source, `Python`-based toolbox for benchmarking multi-component signal analysis methods, implemented either in `Python` or `MATLAB`/`Octave`.

This toolbox provides a common framework that allows researcher-independent comparisons between methods and favors reproducible research.

Create your own collaborative benchmarks using `mcsm-benchs` and this [GitHub template](https://github.com/jmiramont/collab-benchmark-template).

Collaborative benchmarks allows other researchers to add new methods to your benchmark via a `pull-request`. 
This is as easy as creating a new `.py` file with a `Python` class that wrapps a call to your method (it doesn't matter if it is coded in `Python`, `MATLAB` or `Octave`... we wellcome all!). 
**Template files are available** for this too. Let's make collaborative science easy :).

The GitHub workflows provided in the template can automatically publish a summary report [like this](https://jmiramont.github.io/benchmarks-detection-denoising/results_denoising.html) of the benchmarks saved in your repository, as well as make interactive online plots and give access to `.csv` files with the data.

>[!TIP]
>:construction: Questions or difficulties using `mcsm-benchs`?
>
> Please consider leaving [an Issue](https://github.com/jmiramont/mcsm-benchs/issues) so that we can help you and improve our software :white_check_mark:.

## Installation using ```pip```
```bash
pip install mcsm-benchs
```

## Documentation

[![Documentation](docs/readme_figures/docs_badge.svg)](https://jmiramont.github.io/mcsm-benchs)

## Related works

[Work in progress (2024)](https://arxiv.org/abs/2402.08521)

[EUSIPCO 2023](https://github.com/jmiramont/benchmarks_eusipco2023)

[![Gretsi 2022](docs/readme_figures/gretsi_badge.svg)](https://github.com/jmiramont/gretsi_2022_benchmark)

## More

:pushpin: We use [`oct2py`](https://pypi.org/project/oct2py/) to run `Octave`-based methods in `Python`.

:pushpin: We use [`matlabengine`](https://pypi.org/project/matlabengine/) to run `MATLAB`-based methods in `Python`.

:pushpin: We use [`plotly`](https://plotly.com/) to create online, interactive plots.
