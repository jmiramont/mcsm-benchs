---
title: 'mcsm-benchs: Benchmarking statistical methods for multi-component signal processing'

tags:
  - Python
  - Multi-component signals
  - Signal processing
  - Time-frequency representations
  - 
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
Time-frequency (TF) representations are natural encodings of non-stationary time series, also termed signals, helping to discern patterns that reveal their time-varying frequency structure [flandrin1998time].[@gaia]
The model one usually has in mind, when discussing TF representations is a so-called multi-component signal (MCS), where the latter is thought to be the sum of several components of individual interest.
Processing MCSs is an area with a long history which, however, remains a field of methodological innovation.
In such context, systematically comparing existing methods while keeping a record of how they performed on a predefined set of representative problems, i.e. benchmarking, can shed light on novel avenues of research and set clear baselines for new approaches.
This is a widely adopted strategy in neighboring fields such as optimization [hansen2021coco] [benchopt] and machine learning [mattson2020mlperf], yet there are no benchmarking tools for MCSs processing.
`mcsm-benchs` is an open-source Python library for creating reproducible benchmarks of methods that aim at extracting information from MCSs.
`Benchmark` objects can be created by simply passing a series of simulation parameters, a dictionary of methods and a set of performance metrics, while a `SignalBank` class provides several MCSs to standardize comparisons.
Additionally, `mcsm-benchs` includes a `ResultsInterpreter` class that can produce human-readable reports and which functionality underpins *collaborative* benchmarks.
These can be made available through online repositories and can be periodically updated by members of the research community, fostering open and collaborative science.

<!-- Time-frequency (TF) representations are natural encodings of non-stationary time series, or \emph{signals}, helping to discern patterns that reveal their dynamical TF structure \citep{flandrin1998time}. -->
<!-- The signal model one usually has in mind, when discussing TF representations, is a so-called \emph{multi-component signal} (MCS), where the signal is thought to be the sum of several components of individual interest called \emph{modes}. -->
<!-- Additionally, the frequency of each mode can vary with time \citep{auger2013}. -->
<!-- A good TF representation should allow detecting and inferring the characteristics of the different modes, even in the presence of noise. -->
<!-- Multi-component signal processing based on TF representations, particularly on the well-known spectrogram \citep{flandrin1998time}, include the study of several high-level, complex problems like source separation \citep{klejsa2023distribution,sawada2019}, speech enhancement \citep{michelsanti2021overview} and music information processing and retrieval \citep{simonetta2019multimodal}.  -->
<!-- Methodological research also focuses on more elementary tasks, such as detecting an unknown signal in noise, \citep{bardenet2018zeros, bardenet2021time, ghosh2022signal, pascal2022covariant, pascalfamille, miramont2023unsupervised} and estimating the signal (or its components) under stationary noise, \citep{meignen2016adaptive, harmouche2017sliding, meignen2018retrieval, laurent2020novel, colominas2020fully, laurent2021novel, legros2021novel, legros2022pb, ghosh2022signal, legros2022instantaneous, legros2022time}. -->
<!-- New algorithms that tackle these tasks keep appearing.  -->
<!-- For instance, signal detection and denoising methods based on the \emph{zeros} of the spectrogram \citep{flandrin2015time, bardenet2018zeros, ghosh2022signal, pascal2022covariant,miramont2023unsupervised} have been proposed recently. -->
<!-- More practical aspects of the use of spectrogram zeros and their applications in signal detection and denoising are yet to be studied.  -->

# Statement of need

<!-- Thoroughly exploring the parameter space of new methods can be difficult and time-consuming.
Researchers can therefore be surprised at the performance of existing approaches when parameters different from those of the original publication are considered. -->
`mcsm-benchs` brings to the table a common framework to easily carry out extensive comparisons between MCSs based approaches in a unified and objective way.
For instance, `mcsm-benchs` was used to compare statistical methods for signal detection [miramont2022gretsi] and signal denoising of synthetic and real signals under different scenarios, such as white noise and real-world noise [@miramont2024]
Additionally, benchmarks of instantaneous frequency estimation methods where introduced in [miramont2023eusipco].
<!-- What about public benchmarks? -->
While `mcsm-benchs` was mainly designed with MCSs in mind, since methods are always treated like black boxes, i.e. only the shape of the input and output arguments are constrained by the toolbox.
`mcsm-benchs` and can potentially be used to compare any signal processing algorithm as long as its output matches the inputs of a given performance metric.

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