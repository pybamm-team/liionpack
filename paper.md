---
title: 'PoreSpy: A Python Toolkit for Quantitative Analysis of Porous Media Images'

tags:
  - Python
  - porous media
  - tomography
  - image analysis

authors:
  - name: Jeff T. Gostick
    orcid: 0000-0001-7736-7124
    affiliation: 1
  - name: Zohaib A. Khan
    orcid: 0000-0003-2115-7798
    affiliation: 1
  - name: Thomas G. Tranter
    orcid: 0000-0003-4721-5941
    affiliation: "1, 2"
  - name: Matthew D.R. Kok
    orcid: 0000-0001-8410-9748
    affiliation: "2, 3"
  - name: Mehrez Agnaou
    orcid: 0000-0002-6635-080X
    affiliation: 1
  - name: Mohammadamin Sadeghi
    orcid: 0000-0002-6756-9117
    affiliation: 3
  - name: Rhodri Jervis
    orcid: 0000-0003-2784-7802
    affiliation: 2

affiliations:
 - name: Department of Chemical Engineering, University of Waterloo, Waterloo, ON, Canada
   index: 1
 - name: Department of Chemical Engineering, University College London, London, United Kingdom
   index: 2
 - name: Department of Chemical Engineering, McGill University, Montreal, QC, Canada
   index: 3

date: 14 April 2019

bibliography: paper.bib

---

# Summary

Electrification of transport and other energy intensive activities is of growing importance as it provides an underpinning method to reduce carbon emissions. With an increase in reliance on renewable sources of energy and a reduction in the use of more predictable fossil fuels in both stationary and mobile applications, energy storage will play a pivotal role and batteries are currently the most widely adopted and versatile form. Therefore, understanding how batteries work, how they degrade and how to optimize and manage their operation at large scales is critical to achieving emission reduction targets. The electric vehicle (EV) industry requires a considerable number of batteries even for a single vehicle, sometimes numbering in the thousands if smaller cells are used, and the dynamics and degradation of these systems as well as large stationary power systems is not that well understood. As increases in the efficiency of a single battery become diminishing for standard commercially available chemistries, gains made at the system level become more important and can potentially be realised more quickly compared with developing new chemistries. Mathematical models and simulations provide a way to address these challenging questions and can aid the engineer and designers of batteries and battery management systems to provide longer lasting and more efficient energy storage systems.

# Statement of need

`Liionpack` is a PyBaMM-affiliated Python package for simulating large systems of batteries connected in series and parallel. Python enables wrapping low-level languages (e.g., C) for speed without losing flexibility or ease-of-use in the user-interface. The API for `Liionpack` was designed to provide a simple and efficient extension to the `PyBaMM` framework allowing users to scale up simulations from single cells to many thousands with a few extra lines of code. `PyBaMM` provides a number of classic physics-based single battery models with configurable options to investigate thermal effects and degradation for example. The pack architecture introduced by `Liionpack` can be defined as a number of batteries connected in series and parallel to one another using busbars and interconnections with defined resistances. A netlist may also be used to construct the pack which is more flexible and allows for configurable network topology and can be constructed graphically with packages such as `LTSpice` or simply created manually specifying nodal connections as either current sources, voltage sources or resistors. Statistical distributions can be easily incorporated into the pack architecture elements through the use of input parameters that allow a single model to be solved with varying inputs.
<<<<<<< Updated upstream

`Liionpack` was designed to be used by physicists, engineers, students, academics and industrial researchers and system designers concerned with the dynamics of electric current and heat transport in large battery systems. The nature of the solving process facilitates parallel processing of the electrochemical problem formulated as a 1D DAE. Several distributed solvers are provided and can be selected through a common function with a simple function argument. These are `Casadi` which uses multi-threading and works well for single or multi-core machines and `ray` and `dask` [@dask] which are designed for running on clusters and use multi-processing. Many of the functions and models that can be found in `PyBaMM` should work in exactly the same way in `Liionpack` and examples are provided showing how to set up and configure different battery models for running in the pack system. Several visualization tools are also provided for analysis of the results.

# Mathematics

At present, the circuits may only contain three different types of element, namely a current source, voltage source and resistor. Resistors are used to represent the busbars and interconnections in the pack as well as the internal resistance of the batteries. The open circuit voltage is used for the voltage sources in the circuit and modified nodal analysis (MNA) is used to solve the circuit problem determining the distribution of current in the pack.

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.
=======

`Liionpack` was designed to be used by physicists, engineers, students, academics and industrial researchers and system designers concerned with the dynamics of electric current and heat transport in large battery systems. The nature of the solving process facilitates parallel processing of the electrochemical problem formulated as a 1D DAE. Several distributed solvers are provided and can be selected through a common function with a simple function argument. These are `Casadi` which uses multi-threading and works well for single or multi-core machines and `ray` and `dask` [@dask] which are designed for running on clusters and use multi-processing. Many of the functions and models that can be found in `PyBaMM` should work in exactly the same way in `Liionpack` and examples are provided showing how to set up and configure different battery models for running in the pack system. Several visualization tools are also provided for analysis of the results.

# Mathematics

At present, the circuits may only contain three different types of element, namely a current source, voltage source and resistor. Resistors are used to represent the busbars and interconnections in the pack as well as the internal resistance of the batteries. The open circuit voltage is used for the voltage sources in the circuit and modified nodal analysis (MNA) is used to solve the circuit problem determining the distribution of current in the pack.

# Citations

None

# Figures

None

# Acknowledgements

PyBaMM-team acknowledges the funding and support of the Faraday Institution's multi-scale modelling project and Innovate UK.

The development work carried out by members at Oak Ridge National Laboratory was partially sponsored by the Office of Electricity under the United States Department of Energy (DOE).

# References
