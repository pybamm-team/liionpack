[![Python application](https://github.com/pybamm-team/liionpack/actions/workflows/python-app.yml/badge.svg)](https://github.com/pybamm-team/liionpack/actions/workflows/python-app.yml)
[![Documentation Status](https://readthedocs.org/projects/liionpack/badge/?version=latest)](https://liionpack.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/pybamm-team/liionpack/branch/main/graph/badge.svg)](https://codecov.io/gh/pybamm-team/liionpack)

# Overview of liionpack
*liionpack* takes a 1D PyBaMM model and makes it into a pack. You can either specify
the configuration e.g. 16 cells in parallel and 2 in series (16p2s) or load a
netlist

## Installation

Follow the steps given below to install the `liionpack` Python package. The package must be installed to run the included examples. It is recommended to create a virtual environment for the installation.

```bash
# Clone the repository
$ git clone https://github.com/pybamm-team/liionpack.git

# Create a virtual environment in the repository directory
$ cd liionpack
$ python -m venv .venv

# Activate the virtual environment and upgrade pip if venv installed an old version
$ source .venv/bin/activate
$ pip install --upgrade pip

# Install the required packages
$ pip install -r requirements.txt

# Install the liionpack package from within the repository
$ pip install -e .
```

Alternatively, use Conda to create a virtual environment then install the `liionpack`  package.

```bash
# Clone the repository
$ git clone https://github.com/pybamm-team/liionpack.git

# Create a Conda virtual environment
$ cd liionpack
$ conda env create -f environment.yml

# Activate the conda environment
$ conda activate lipack

# Install the liionpack package from within the repository
$ pip install -e .
```

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

# Cycling protocol
protocol = lp.generate_protocol()

# PyBaMM parameters
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

# Solve pack
output = lp.solve(netlist=netlist,
                  parameter_values=parameter_values,
                  protocol=protocol,
                  output_variables=output_variables,
                  htc=htc)
```

## Acknowledgments
PyBaMM-team acknowledges the funding and support of the Faraday Institution's multi-scale modelling project and Innovate UK. 

The development work carried out by members at Oak Ridge National Laboratory was partially sponsored by the Office of Electricity under the United States Department of Energy (DOE).
