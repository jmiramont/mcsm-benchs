# What is `mcsm-benchs`?

`mcsm-benchs` is a Python-based toolbox for benchmarking methods that aim at detecting, analyzing, and processing signals that are the addition of several individual components of interest, i.e. multi-component signals.

This toolbox was born out of the necessity of simplifying systematic comparison of new approaches with existing ones in an area with vast amounts of methods like time-frequency representations of multi-component signals, which still remains a field of methodological innovation.
Tools with similar goals are already being used in neighboring fields like optimization or machine learning.

`mcsm-benchs` was designed to underpin collaborative benchmarks available through online repositories (see here a custom GitHub [template](https://github.com/jmiramont/collab-benchmark-template) to create your own).

# Architecture and basic functionality
`mcsm-benchs` was designed as a modular software, so that different modules can interact with each other. They may also be easily updated, or replaced, without affecting the other components in a benchmark.
The object-oriented programming paradigm was extensively used, and the basic components of a benchmark, as well as other utilities provided by the library, are represented by different `Python` classes.


```{figure} ./readme_figures/dessin_color_2.svg
:alt: Block diagram describing `mcsm-benchs`classes and their interaction.
:name: block-diagram
:width: 700px
:align: center
Block diagram describing `mcsm-benchs`classes and their interaction.
```

Figure {numref}`block-diagram` shows a block diagram depicting the interactions between the main components of the toolbox and the inputs from the user.
Optional input parameters are denoted with dotted lines.
The three main classes of `mcsm-benchs` are: 1) the `SignalBank`class, 2) the `Benchmark` class and 3) the `ResultsInterpreter` class.
Figure {numref}`block-diagram` also shows the four main attributes of the `Benchmark` class: 
1. the noise generation function.
2. the dictionary of methods to be compared.
3. the dictionary of performance metrics
4. the results.

Following the pipeline in {numref}`block-diagram`, the user defines a set of signals to test, or generates them using the methods provided by the `SignalBank`class.
Then, the simulation parameters, such as the length of the signals, the number of noise realizations to use, etc., are passed to the `Benchmark` class creator and a `Benchmark` object is instantiated.

To run the benchmark, the class method [`run(...)`](https://jmiramont.github.io/mcsm-benchs/mcsm_benchs.html#mcsm_benchs.Benchmark.Benchmark.run) is called, which first executes the noise generation function to create a set of noise realizations.
Then, these realizations are added to the signals provided by the user, yielding signal-plus-noise mixtures with the specified signal-to-noise ratios (SNR, in dB).
The noisy signals are then fed to the methods provided by the user as function handlers in a `Python` dictionary of methods.
Their outputs are in turn given to the performance metrics.
The main final product of a benchmark created with `mcsm-benchs` is a two-dimensional data array, technically a Pandas `DataFrame`.
Results can be processed by the `ResultsInterpreter` class, which generates output files and interactive figures to show in custom websites.

**If a method raises an exception during execution time, the benchmarking process is not stopped.**
Instead, the exception is caught internally, and a warning is shown in the standard output, indicating that the results of that iteration are replaced by `NaN`.
Additionally, a log entry is saved with the method name and the noise realization that was used in that iteration.
The log is a `Benchmark` attribute, and can then be helpful to reproduce any particular error during the benchmarking process.

## Noise generation function

A function handler to a noise generator method (or a dictionary of thse) can be passed by the user as an input parameter.
The signature of the functions should be 

```python
def noise_fun(N):
  ...
```

where `N` is an integer defining the length of the signals (in time samples).

The value of `N` is passed by the `Benchmark` object to the `noise_fun(...)` function during the execution of the `Benchmark` class method `run(...)`.
By default, the benchmarks use white Gaussian noise (either complex- or real-valued according to the signal).

## Dictionary of methods

In order to make the toolbox versatile, a method is considered as a generic `Python` function with a variable number of input arguments using the following signature:

```python
def method(noisy_signal, *args, **kwargs):
  ...
```

where `noisy_signal` is a one-dimensional `numpy` array.
The `*args` and `**kwargs` arguments represent a variable number of positional or keyword arguments that can be passed to parametrize the method.
It is up to the user to define what the function `method(...)` does, and to set up any number of parameter combinations that could be fed to the method.

The function handlers of the methods are then assigned to keys in a `Python` dictionary created by the user.
Such keys are also used to identify the method in the tables and reports produced later.
The only limitation imposed by the toolbox on `method(...)` is the shape of the output, which depends on the task. 
If `task='denoising'`, the output must have the same dimensions as `noisy_signal`. 
For `task='detection'`, the output must be a Boolean variable, indicating whether a signal has been detected or not.
For `task='misc'`, the shape of the output is not limited, but it must be handled by a performance metric provided by the user, as explained in the following.

## Performance metrics

Performance metrics are computed by generic functions that receive a vector array with the original signal `x`, the output of a method `y`, and a variable number of input parameters `**kwargs`:

```python
def performance_metric(x, y, **kwargs):
  ...
```

The variable number of keyword arguments is used to receive further information from the `Benchmark` class which might be needed to compute the performance metrics.
An example of such information could be the noise realizations used to contaminate `x`.
This provides users with the flexibility needed to implement their own performance functions.
In addition, several performance metrics may be employed by passing function handlers to compute them contained in a `Python` dictionary.
The keys of such dictionary are used then as identifiers of the corresponding performance metrics when results are tabulated.

# The `SignalBank` class

```{figure} ./readme_figures/figure_signals_example.svg
:alt: Spectrograms of some of the signals generated by the `SignalBank`class.
:name: spectrograms
:width: 700px
:align: center
Spectrograms of some of the signals generated by the `SignalBank`class.
```

Noisy multi-component synthetic signals enable quantitative evaluation of the performance of a method, since both the noiseless version of the signal and the noise are known.
The `SignalBank`class included in `mcsm-benchs` can be used to generate more than $20$ multi-component signals with a variety of TF structures, which serves the purpose of standardizing benchmarks across different users by providing a common input to the methods.
{numref}`spectrograms` displays the spectrograms of some of the available signals produced by the `SignalBank` class.

More specifically, a discrete multi-component, amplitude-modulated, frequency-modulated (AM-FM) signal $s$ can be written as

```{math}
s[n] = \sum_{j=1}^{J} s_{j}[n], \text{ with } s_{j}[n] = a_{j}[n]e^{i \phi_{j}[n]}
```

where $n=0,1,\dots,N-1$ is the discrete time index, $J$ is the number of signal components, $s_{j}$ is an AM-FM component, and $a_{j}$ and $\phi_{j}$ are the instantaneous amplitude and phase functions of the $j$-th mode, respectively.
Each mode is associated with an instantaneous frequency given by $\frac{1}{2\pi}\varphi^{\prime}(nT_{s})$, $n=0,1,\dots,N-1$, where $\varphi$ is the continuous-time counterpart of the phase function corresponding to the $j$-$th$ mode and $T_{s}$ is the sampling period.

Multi-component signals can then be designed to pose specific challenges to methods, which is relevant when the approaches to be benchmarked are based on *models* of either the signal, the noise, or both of them.
For instance, multi-component signals can be comprised of impulse-like fast transients, several kinds of chirps obeying different instantaneous frequency laws, components that born and die at different instants throughout the time span of the signal, etc.

% The `SignalBank`class offers more than $20$ multi-component synthetic signals with different time-frequency structures.
% Detecting the ridges is the cornerstone of the methods based on the largest values of the spectrogram.

A `SignalBank`object outputs signals as regular `numpy` arrays or as instances of a custom `Signal` class. 
In practice, `Signal` objects behave like regular `numpy` arrays, but they also store useful multi-component signal attributes like:

- `ncomps`: `numpy` array with the number of components at each time instant.
- `total_comps`: Integer indicating the total number of signal components.
- `comps`: `numpy` array with the individual components (as independent signals).
- `instf`: `numpy` array with the instantaneous frequency of each component.

{numref}`signal-attributes` illustrates the attributes of a `Signal` class instance: the signal, its individual components as well as their instantaneous frequencies, and the number of components present in the signal for each instant.

```{figure} ./readme_figures/signals_ifs.svg
:alt: Attributes of the `Signal`class.
:name: signal-attributes
:width: 700px
:align: center
Attributes of the `Signal`class. First row, left: Spectrogram of a synthetic signal with four components.  Second row, left: Number of components present in the signal at each time instant. The four remaining plots show the spectrograms of the individual components with their respective instantaneous frequencies superimposed (in dashed red lines). 
```

The `SignalBank`and `Signal` classes are a core feature of `mcsm-benchs` that makes it useful for evaluating the performance of MCS-based methods.

For instance, the attributes of a `Signal` object can be queried by the methods and the performance metrics when running the benchmarks.
This is quite useful to compare not only signal detection or denoising methods but also approaches that estimate individual components or their instantaneous frequencies.
Approaches like those described in {cite}`legros2021novel, harmouche2017sliding` require knowing the number of components, or the number of components for each time instant.
These can be easily obtained from a `Signal` class instance, and all these multi-component signal oriented methods can be benchmarked using `mcsm-benchs`.
Examples of such applications are introduced in {cite}`miramont2022public, miramont2023eusipco, miramont2024`.


# The `ResultsInterpreter` class

Processing benchmark data can be a cumbersome task. 
For this reason, `mcsm-benchs` includes a `ResultsInterpreter` class that provides tools to easily process and publish results in a similar format regardless of the user.

An instance of `ResultsInterpreter` is created by passing a `Benchmark` object to the class constructor, after which a variety of class methods can be used to generate reports, figures and files from the output of the performance metrics.

As an example, the figures later discussed in Sec. \ref{sec:results} were generated using the class method `get_summary_plots(...)`.
The function `save_report(...)` can be used to generate reports comprising the simulation parameters and a summary of the results.
Reports generated using `ResultsInterpreter` class include links to download comma-separated values (\code{.csv}) files with the results and access to interactive plots created by the function `get_summary_plotlys(...)` using `Plotly`.
The main goal of these reports is to be published on-line and shared with users to foster collaboration for more details on collaborative benchmarks.

Confidence intervals (CI) are also shown in the plots produced by `get_summary_plotlys(...)` and `get_summary_plots(...)`.
For `task='misc'` and `task='denoising'`, the $95\%$ boostrap CIs are computed using `Seaborn` {cite}`Waskom2021`.
For `task='detection'` the output of each method is a boolean, hence 95\% Clopper-Pearson CIs are used.
In either case, Bonferroni correction is automatically applied to account for the comparison of multiple methods for each SNR.

# Extra functionalities
Some extra functionalities that can be of interest for users of `mcsm-benchs` are summarized below:
- Methods can be run in parallel when appropriate hardware is available, reducing the benchmarking time. This can be indicated by the user as an input parameter, as well as the desired number of processes.
- The execution time of each method is automatically computed, allowing users to take it into account in their comparisons (not available in parallel computation).
- `mcsm-benchs` provides a custom `MatlabInterface` class based on  `MATLAB`'s own `matlabengine` module, that allows straightforward use of methods implemented in `MATLAB` directly from the `Python`-based benchmark. Restrictions on `MATLAB`'s version may apply in this case. Similarly, a custom `OctaveInterface` class based on `oct2py` is provided to support methods implemented in `Octave`. Examples of use are given in the library documentation.
- Interactive figures can be easily obtained from the `ResultsInterpreter` class, and added to online repositories.
- Even though using synthetic signals from the `SignalBank`class can help standardize benchmarks created with `mcsm-benchs`, users are not limited to the signals synthesized by this class.
Other synthetic or real-world signals can be used when creating new benchmarks {cite}`miramont2023eusipco, miramont2024`.

## References

```{bibliography}
```
