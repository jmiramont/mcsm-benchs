[![Tests](https://github.com/jmiramont/mcsm-benchs/actions/workflows/tests.yml/badge.svg)](https://github.com/jmiramont/mcsm-benchs/actions/workflows/tests.yml) [![codecov](https://codecov.io/gh/jmiramont/mcsm-benchs/graph/badge.svg?token=CJPPKYJD8H)](https://codecov.io/gh/jmiramont/mcsm-benchs) [![Documentation](docs/readme_figures/docs_badge.svg)](https://jmiramont.github.io/mcsm-benchs)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# `mcsm-benchs`: A Toolbox for Benchmarking Multi-Component Signal Analysis Methods

A public, open-source, `Python`-based toolbox for benchmarking multi-component signal analysis methods, implemented either in `Python` or `MATLAB`/`Octave`.

This toolbox provides a common framework that allows researcher-independent comparisons between methods and favors reproducible research.

Create your own collaborative benchmarks using `mcsm-benchs` and this [GitHub template](https://github.com/jmiramont/collab-benchmark-template).

Collaborative benchmarks allow other researchers to add new methods to your benchmark via a `pull-request`.
This is as easy as creating a new `.py` file with a `Python` class that wraps a call to your method (it doesn't matter if it is coded in `Python`, `MATLAB` or `Octave`... we welcome all!).
[**Template files are available**](https://github.com/jmiramont/collab-benchmark-template/tree/main/new_method_examples) for this too. Let's make collaborative science easy :).

The GitHub workflows provided in the template can automatically publish a summary report [like this](https://jmiramont.github.io/benchmarks-detection-denoising/results_denoising.html) of the benchmarks saved in your repository, as well as make interactive online plots and give access to `.csv` files with the data.

>[!TIP]
>:construction: Questions or difficulties using `mcsm-benchs`?
>
> Please consider leaving [an Issue](https://github.com/jmiramont/mcsm-benchs/issues) so that we can help you and improve our software :white_check_mark:.

>[!TIP]
> Do you want to contribute to `mcsm-benchs`?
>
> Please check our [contribution guidelines]() :white_check_mark:.



## Installation using ```pip```

```bash
pip install mcsm-benchs
```

For installation in development mode using `poetry` [check instructions in the documentations](https://jmiramont.github.io/mcsm-benchs/install.html)

## Documentation

[![Documentation](docs/readme_figures/docs_badge.svg)](https://jmiramont.github.io/mcsm-benchs)

## Quick-start

### Creating a new benchmark

The following simple example shows how to create a new benchmark for comparing your methods.
We set `task=denoising`, meaning that all methods will be compared in terms of reconstruction of the original signal from noise.

Check out examples with other tasks and performance functions in the [documentation](https://jmiramont.github.io/mcsm-benchs/) of `mcsm-benchs`.

```python
from mcsm_benchs.Benchmark import Benchmark
from mcsm_benchs.SignalBank import SignalBank
# 1. Import (or define) the methods to be compared.
from my_methods import method_1, method_2

# 2. Create a dictionary of the methods to test.
my_methods = { 'Method 1': method_1, 'Method 2': method_2, }

# 3. Create a dictionary of signals:
N = 1024                                    # Length of the signals
sbank = SignalBank(N,)
s1 = sbank.signal_exp_chirp()
s2 = sbank.signal_linear_chirp()
my_signals = {'Signal_1':s1, 'Signal_2':s2, }

# 4. Set the benchmark parameters:
benchmark = Benchmark(task='denoising',
                    N=N, 
                    repetitions = 100,
                    SNRin=[0,10,20],        # SNR in dB.
                    methods=my_methods, 
                    signals=my_signals,
                    verbosity=0
                    )
# 5. Launch the benchmark and save to file
benchmark.run()                        # Run the benchmark.
benchmark.save_to_file('saved_benchmark')   # Give a filename and save to file
```

### Processing and publishing benchmark results

```python
from mcsm_benchs.Benchmark import Benchmark
from mcsm_benchs.ResultsInterpreter import ResultsInterpreter

# 1. Load a benchmark from a file.
benchmark = Benchmark.load('path/to/file/saved_benchmark')

# 2. Create the interpreter
interpreter = ResultsInterpreter(benchmark)

# 3. Get .csv files
interpreter.get_csv_files(path='path/to/csv/files')

# 4. Generate report and interactive figures
interpreter.save_report(path='path/to/report', bars=False)

#5 Or generate interactive plots with plotly
from plotly.offline import iplot
figs = interpreter.get_summary_plotlys(bars=True)
for fig in figs:
    iplot(fig)
```

If you use the GitHub [template for collaborative benchmarks](https://github.com/jmiramont/collab-benchmark-template), your results are automatically published if you enable GitHub sites in the repository configuration.
Additionally, other researchers will be able to interact with your results, download `.csv` files with all the benchmark data and even add their own methods to your benchmark via a *pull-request*.

## Related works

[Work in progress (2024)](https://arxiv.org/abs/2402.08521)

[EUSIPCO 2023](https://github.com/jmiramont/benchmarks_eusipco2023)

[![Gretsi 2022](docs/readme_figures/gretsi_badge.svg)](https://github.com/jmiramont/gretsi_2022_benchmark)

## More

:pushpin: We use [`oct2py`](https://pypi.org/project/oct2py/) to run `Octave`-based methods in `Python`.

:pushpin: We use [`matlabengine`](https://pypi.org/project/matlabengine/) to run `MATLAB`-based methods in `Python`.

:pushpin: We use [`plotly`](https://plotly.com/) to create online, interactive plots.