---
title: 'mcsm-benchs: Benchmarking methods for multi-component signal processing'

tags:
  - Python
  - Multi-component signals
  - Signal processing
  - Time-frequency representations
  
authors:
  - name: Juan M. Miramont
    orcid: 0000-0002-3847-7811
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Rémi Bardenet
    orcid: 0000-0002-1094-9493
    affiliation: 1
  - name: Pierre Chainais
    orcid: 0000-0003-4377-7584
    affiliation: 1
  - name: François Auger
    orcid: 0000-0001-9158-1784
    affiliation: 2

affiliations:
 - name: Université de Lille, CNRS, Centrale Lille, UMR 9189 Centre de Recherche en Informatique, Signal et Automatique de Lille (CRIStAL), Lille, France.
   index: 1
 - name: Nantes Université, Institut de Recherche en Énergie Électrique de Nantes Atlantique (IREENA, UR 4642), Saint-Nazaire, France.
   index: 2

date: 26 January 2025
bibliography: paper.bib
---

# Summary

Time-frequency (TF) representations are natural encodings of non-stationary time series, also termed signals, helping to discern patterns that reveal their time-varying frequency structure [@flandrin1998time].
The model one usually has in mind, when discussing TF representations is a so-called multi-component signal (MCS), where the latter is thought to be the sum of several components of individual interest.

`mcsm-benchs` is an open-source Python library for creating reproducible benchmarks of methods that aim at extracting information from MCSs.

`Benchmark` objects can be created by simply passing a series of simulation parameters, a dictionary of methods and a set of performance metrics, while a `SignalBank` instance can be used to standardize comparisons across benchmarks.
The `SignalBank` class can synthesize several MCSs as objects from a custom `Signal` class that behave like regular arrays in Python but also contain information about the components.

Additionally, `mcsm-benchs` includes a `ResultsInterpreter` class that can produce human-readable reports, the functionality of which underpins *collaborative* benchmarks [@benchopt].
These are based on online repositories (see [here](https://jmiramont.github.io/benchmarks-detection-denoising/results_denoising.html) for an example) and can be periodically updated by members of the research community, fostering open and collaborative science.
Several examples are given in the [documentation](https://jmiramont.github.io/mcsm-benchs/), as well as a GitHub [repository template](https://github.com/jmiramont/collab-benchmark-template) that relies heavily on `mcsm-benchs` and continuous integration/deployment workflows, in order to automatize the process of publishing new benchmarks.

# Statement of need

MCS processing is an area with a long history which, however, remains a field of methodological innovation [@bardenet2018zeros; @colominas2020fully; @kreme2020filtering; @bardenet2021time; @legros2022instantaneous; @legros2022time; @ghosh2022signal; @pascal2022covariant; @pascalfamille].
In such a context, systematically comparing existing methods while keeping a record of how they performed on a predefined set of representative problems, i.e. benchmarking, can shed light on novel avenues of research and set clear baselines for new approaches.
This is a widely adopted strategy in neighboring fields such as optimization [@bartz2020benchmarking; @hansen2021coco; @benchopt] and machine learning [@mattson2020mlperf], yet there are no benchmarking tools for MCS processing.

`mcsm-benchs` brings to the table a common framework to easily carry out extensive comparisons between MCS-based approaches in a unified and objective way.
It can be used to benchmark any number of approaches and create clear baselines for new methods that are accessible to the whole research community.

The toolbox is versatile enough to allow comparisons between many kinds of methods.
For instance, `mcsm-benchs` was used to compare statistical tests for signal detection [@miramont2022public], and for denoising of synthetic and realistic signals under different scenarios, such as white noise or even real-world noises [@miramont2024].
As another example, many approaches within MCS processing focus instead on retrieving individual components and estimating their instantaneous frequencies.
These methods can be easily benchmarked using `mcsm-benchs` as well (see @miramont2023eusipco).

While the aforementioned cases illustrate the most typical applications in MCS processing, methods are in fact always treated like *black boxes* by `mcsm-benchs`.
The only constraint imposed by the software is that the outputs of a method should match the inputs of the performance metrics given by the user.
Thanks to this feature, `mcsm-benchs` has a significant potential, as it could be used to systematically compare any signal processing algorithm.
Large studies of methods for specific applications can thus be created and kept updated by the signal processing community using `mcsm-benchs`, hopefully leading to widely adopted procedures for evaluating new approaches that are transparent, less time-consuming and straightforward for researchers to use.

Finally, in order to ease the adoption of this package by the community, `mcsm-benchs` also supports methods coded in `Octave`/`Matlab`, so that these can be seamlessly integrated into Python-based benchmarks.

# Acknowledgements
This work was supported by grant ERC-2019-STG-851866, and French projects ANR20-CHIA-0002 and ASCETE-ANR19-CE48-0001.
JMM would like to thank Guillaume Gautier and Yusuf Yiğit Pilavcı for their valuable insight.

# References