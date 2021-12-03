---
title: 'Liionpack: A Python package for simulating packs of batteries with PyBaMM'

tags:
  - Python
  - batteries
  - packs
  - electrochemistry

authors:
  - name: T. G. Tranter
    orcid: 0000-0003-4721-5941
    affiliation: "1, 2"
  - name: R. Timms
    orcid: 0000-0002-8858-4818
    affiliation: "2, 3"
  - name: V. Sulzer
    orcid: 0000-0002-8687-327X
    affiliation: "4"
  - name: F. Brosa Planella
    orcid: 0000-0001-6363-2812
    affiliation: "2, 5"
  - name: G. M. Wiggins
    orcid: 0000-0002-4737-6596
    affiliation: "6"
  - name: V. Karra
    orcid: 0000-0002-5671-0998
    affiliation: "6"
  - name: P. Agarwal
    orcid: 0000-0002-5333-1634
    affiliation: "7"
  - name: S. Chopra
    orcid: 0000-0003-3046-7675
    affiliation: "8"
  - name: S. Allu
    orcid: 0000-0003-2841-4398
    affiliation: "6"
  - name: P. Shearing
    orcid: 0000-0002-1387-9531
    affiliation: "1, 2"
  - name: D. J. L. Brett
    orcid: 0000-0002-8545-3126
    affiliation: "1, 2"

affiliations:
 - name: Department of Chemical Engineering, University College London, London, WC1E 7JE, United Kingdom.
   index: 1
 - name: The Faraday Institution, Quad One, Becquerel Avenue, Harwell Campus, Didcot, OX11 0RA, United Kingdom.
   index: 2
 - name: Mathematical Institute, University of Oxford, OX2 6GG, United Kingdom.
   index: 3
 - name: Carnegie Mellon University, Scott Hall 5109, 5000 Forbes Ave, Pittsburgh, PA 15213, United States.
   index: 4
 - name: WMG, University of Warwick, Coventry, CV4 7AL, United Kingdom
   index: 5
 - name: Oak Ridge National Laboratory, 2360 Cherahala Boulevard, Knoxville, Tennessee 37932, United States.
   index: 6
 - name: Symbiosis Institute of Technology, Symbiosis International University, Lavale, Pune, Maharashtra 412115, India.
   index: 7
 - name: Cluster Innovation Centre, University of Delhi, GC Narang Road, Delhi, 110007, India.
   index: 8

date: 03 December 2021

bibliography: paper.bib

---

# Summary

Electrification of transport and other energy intensive activities is of growing importance as it provides an underpinning method to reduce carbon emissions. With an increase in reliance on renewable sources of energy and a reduction in the use of more predictable fossil fuels in both stationary and mobile applications, energy storage will play a pivotal role and batteries are currently the most widely adopted and versatile form. Therefore, understanding how batteries work, how they degrade and how to optimize and manage their operation at large scales is critical to achieving emission reduction targets. The electric vehicle (EV) industry requires a considerable number of batteries even for a single vehicle, sometimes numbering in the thousands if smaller cells are used, and the dynamics and degradation of these systems as well as large stationary power systems is not that well understood. As increases in the efficiency of a single battery become diminishing for standard commercially available chemistries, gains made at the system level become more important and can potentially be realised more quickly compared with developing new chemistries. Mathematical models and simulations provide a way to address these challenging questions and can aid the engineer and designers of batteries and battery management systems to provide longer lasting and more efficient energy storage systems.

# Statement of need

`Liionpack` is a PyBaMM-affiliated Python package for simulating large systems of batteries connected in series and parallel. Python enables wrapping low-level languages (e.g., C) for speed without losing flexibility or ease-of-use in the user-interface. The API for `Liionpack` was designed to provide a simple and efficient extension to the `PyBaMM` [@pybamm] framework allowing users to scale up simulations from single cells to many thousands with a few extra lines of code. `PyBaMM` provides a number of classic physics-based single battery models with configurable options to investigate thermal effects and degradation for example. The pack architecture introduced by `Liionpack` can be defined as a number of batteries connected in series and parallel to one another using busbars and interconnections with defined resistances. A netlist may also be used to construct the pack which is more flexible and allows for configurable network topology and can be constructed graphically with packages such as `LTSpice` [@ltspice] or simply created manually specifying nodal connections as either current sources, voltage sources or resistors. Statistical distributions can be easily incorporated into the pack architecture elements through the use of input parameters that allow a single model to be solved with varying inputs.

`Liionpack` was designed to be used by physicists, engineers, students, academics and industrial researchers and system designers concerned with the dynamics of electric current and heat transport in large battery systems. The nature of the solving process facilitates parallel processing of the electrochemical problem formulated as a 1D DAE. Several distributed solvers are provided and can be selected through a common function with a simple function argument. These are `Casadi` [@casadi] which uses multi-threading and works well for single or multi-core machines and `ray` [@ray] and `dask` [@dask] which are designed for running on clusters and use multi-processing. Many of the functions and models that can be found in `PyBaMM` should work in exactly the same way in `Liionpack` and examples are provided showing how to set up and configure different battery models for running in the pack system. Several visualization tools are also provided for analysis of the results.

# Mathematics

At present, the circuits may only contain three different types of element, namely a current source, voltage source and resistor. Resistors are used to represent the busbars and interconnections in the pack as well as the internal resistance of the batteries. The open circuit voltage is used for the voltage sources in the circuit and modified nodal analysis (MNA) [@mna] is used to solve the circuit problem determining the distribution of current in the pack.

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

None

# Figures

None

# Acknowledgements

PyBaMM-team acknowledges the funding and support of the Faraday Institution's multi-scale modelling project and Innovate UK.

The development work carried out by members at Oak Ridge National Laboratory was partially sponsored by the Office of Electricity under the United States Department of Energy (DOE).

# References
