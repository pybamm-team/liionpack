![logo](https://raw.githubusercontent.com/pybamm-team/liionpack/main/docs/liionpack.png)

#
<div align="center">

[![liionpack](https://github.com/pybamm-team/liionpack/actions/workflows/test_on_push.yml/badge.svg)](https://github.com/pybamm-team/liionpack/actions/workflows/test_on_push.yml)
[![Documentation Status](https://readthedocs.org/projects/liionpack/badge/?version=develop)](https://liionpack.readthedocs.io/en/develop/?badge=develop)
[![codecov](https://codecov.io/gh/pybamm-team/liionpack/branch/main/graph/badge.svg)](https://codecov.io/gh/pybamm-team/liionpack)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pybamm-team/liionpack/blob/main/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04051/status.svg)](https://doi.org/10.21105/joss.04051)

</div>

# Overview of liionpack
*liionpack* takes a 1D PyBaMM model and makes it into a pack. You can either specify
the configuration e.g. 16 cells in parallel and 2 in series (16p2s) or load a
netlist.

## Installation

Follow the steps given below to install `liionpack`. The package must be installed to run the included examples. It is recommended to create a virtual environment for the installation, see [the documentation](https://liionpack.readthedocs.io/en/main/install/).

To install `liionpack` using `pip`, run the following command:
```bash
pip install liionpack
```

### Conda

The following terminal commands are for setting up a conda development environment for liionpack. This requires the [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) Python distribution. This environment installs liionpack in editable mode which is useful for development of the liionpack source code. General users should install liionpack with pip.

```bash
# Create a conda environment named lipack
cd liionpack
conda env create --file environment.yml

# Activate the environment
conda activate lipack

# Exit the environment
conda deactivate

# Delete the environment
conda env remove --name lipack
```

### LaTeX

In order to use the `draw_circuit` functionality a version of Latex must be installed on your machine. We use an underlying Python package `Lcapy` for making the drawing and direct you to its installation instructions [here](https://lcapy.readthedocs.io/en/latest/install.html) for operating system specifics.

## Example Usage

The following code block illustrates how to use liionpack to perform a simulation:

```python
import liionpack as lp
import numpy as np
import pybamm

# Generate the netlist
netlist = lp.setup_circuit(Np=16, Ns=2, Rb=1e-4, Rc=1e-2, Ri=5e-2, V=3.2, I=80.0)

output_variables = [
    'X-averaged total heating [W.m-3]',
    'Volume-averaged cell temperature [K]',
    'X-averaged negative particle surface concentration [mol.m-3]',
    'X-averaged positive particle surface concentration [mol.m-3]',
]

# Heat transfer coefficients
htc = np.ones(32) * 10

# Cycling experiment, using PyBaMM
experiment = pybamm.Experiment([
    "Charge at 20 A for 30 minutes",
    "Rest for 15 minutes",
    "Discharge at 20 A for 30 minutes",
    "Rest for 30 minutes"],
    period="10 seconds")

# PyBaMM parameters
parameter_values = pybamm.ParameterValues("Chen2020")

# Solve pack
output = lp.solve(netlist=netlist,
                  parameter_values=parameter_values,
                  experiment=experiment,
                  output_variables=output_variables,
                  htc=htc)
```

## Documentation

There is a full API documentation, hosted on Read The Docs that can be found [here](https://liionpack.readthedocs.io/).

## Contributing to liionpack

If you'd like to help us develop liionpack by adding new methods, writing documentation, or fixing embarrassing bugs, please have a look at these [guidelines](https://github.com/pybamm-team/liionpack/blob/main/docs/contributing.md) first.

## Get in touch

For any questions, comments, suggestions or bug reports, please see the [contact page](https://www.pybamm.org/contact).

## Acknowledgments

PyBaMM-team acknowledges the funding and support of the Faraday Institution's multi-scale modelling project and Innovate UK.

The development work carried out by members at Oak Ridge National Laboratory was partially sponsored by the Office of Electricity under the United States Department of Energy (DOE).

## License

liionpack is fully open source. For more information about its license, see [LICENSE](https://github.com/pybamm-team/liionpack/blob/main/LICENSE).
