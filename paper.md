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
 - name: Université de Lille, CNRS, Centrale Lille, UMR 9189 Centre de Recherche en Informatique, Signal et Automatique de Lille (CRIStAL), F-59000, Lille, France.
   index: 1
 - name: Nantes Université, Institut de Recherche en Énergie Électrique de Nantes Atlantique (IREENA, UR 4642), CRTT, 37 Boulevard de l'Université, CS 90406, F-44612 Saint-Nazaire, France.
   index: 2

date: 26 January 2025
bibliography: paper.bib
---

# Summary

Time-frequency (TF) representations are natural encodings of non-stationary time series, also termed signals, helping to discern patterns that reveal their time-varying frequency structure [@flandrin1998time].
The model one usually has in mind, when discussing TF representations is a so-called multi-component signal (MCS), where the latter is thought to be the sum of several components of individual interest.

MCS processing is an area with a long history which, however, remains a field of methodological innovation [@bardenet2018zeros], [@colominas2020fully], [@kreme2020filtering], [@bardenet2021time], [@legros2022instantaneous], [@legros2022time],[@ghosh2022signal], [@pascal2022covariant], [@pascalfamille].
In such a context, systematically comparing existing methods while keeping a record of how they performed on a predefined set of representative problems, i.e. benchmarking, can shed light on novel avenues of research and set clear baselines for new approaches.
This is a widely adopted strategy in neighboring fields such as optimization [@bartz2020benchmarking] [@hansen2021coco], [@benchopt] and machine learning [@mattson2020mlperf], yet there are no benchmarking tools for MCS processing.
`mcsm-benchs` is an open-source Python library for creating reproducible benchmarks of methods that aim at extracting information from MCSs.
`Benchmark` objects can be created by simply passing a series of simulation parameters, a dictionary of methods and a set of performance metrics, while a `SignalBank` can be used to standardize comparisons across benchmarks.
This latter can synthesize several MCSs as objects from a custom `Signal` class that behave like regular arrays in Python but also contain information about the components.

Additionally, `mcsm-benchs` includes a `ResultsInterpreter` class that can produce human-readable reports, the functionality of which underpins *collaborative* benchmarks.
These are based on online repositories (see [here](https://jmiramont.github.io/benchmarks-detection-denoising/results_denoising.html) for an example) and can be periodically updated by members of the research community, fostering open and collaborative science.
Several examples are given in the documentation, as well as a GitHub [repository template](https://github.com/jmiramont/collab-benchmark-template) that relies heavily on `mcsm-benchs` and continuous integration/deployment workflows, in order to automatize the process of publishing new benchmarks.

# Statement of need

`mcsm-benchs` brings to the table a common framework to easily carry out extensive comparisons between MCS-based approaches in a unified and objective way.
For instance, `mcsm-benchs` was used to compare statistical methods for signal detection [@miramont2022public] and signal denoising of synthetic and real signals under different scenarios, such as white noise and real-world noises [@miramont2024].
Additionally, benchmarks of instantaneous frequency and component estimation methods where introduced in [@miramont2023eusipco].
<!-- What about public benchmarks? -->
While `mcsm-benchs` was mainly designed with MCSs in mind, methods are always treated like black boxes, i.e. only the shape of the input and output arguments are constrained by the toolbox.
Therefore, `mcsm-benchs` can potentially be used to compare any signal processing algorithm as long as its output matches the inputs of a given performance metric.
Finally, to ease the adoption of our tool by the community, `mcsm-benchs` also supports methods coded in `Octave`/`Matlab`, so that these can be seamlessly integrated into new Python-based benchmarks.

<!-- # Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commandscite can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

<!-- # Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements
This work was supported by grant ERC-2019-STG-851866, and French projects ANR20-CHIA-0002 and ASCETE-ANR19-CE48-0001.
JMM would like to thank Yusuf Yiğit Pilavcı for his valuable insight.

# References