# [Unreleased](https://github.com/pybamm-team/liionpack/)

# [v0.3.1](https://github.com/pybamm-team/PyBaMM/tree/v0.3.1) - 2022-05-24

## Features

-   Add functions for saving simulation output to csv and npz ([#145](https://github.com/pybamm-team/liionpack/pull/145))
-   Internally change generation of ParameterValues to new PyBaMM format ([#134](https://github.com/pybamm-team/liionpack/pull/134))
-   Change model timescale to a scalar so that scaling is consistent in batteries of different sizes for single model. Add external thermal simulation function. Update circuit solve vectorized to work with circuit topology with multiple similar resistors connected to single nodes. ([#124](https://github.com/pybamm-team/liionpack/pull/124))

## Bug fixes

-   Fix build, update solution inistialisation ([#148](https://github.com/pybamm-team/liionpack/pull/148))
-   Add jax install to github actions ([#154](https://github.com/pybamm-team/liionpack/pull/154))

## Breaking changes

-   Change solver class names to camelcase. Does not break usage if using wrapper solve functions ([#132](https://github.com/pybamm-team/liionpack/pull/132))

# [v0.3](https://github.com/pybamm-team/PyBaMM/tree/v0.3) - 2022-02-17
This is the first official version of liionpack.
Please note that liionpack and PyBaMM are both still under active development, and so the API may change in the future.

## Features

- Define a pack architecture with number of batteries in series and / parallel
- Load a pack architecture from a SPICE netlist
- Run pack simulations using PyBaMM simulations and experiments
- Lumped thermal model with configurable heat transfer coefficients
- Implement cell variablity with input parameters
- Run large simulations with parallel processing