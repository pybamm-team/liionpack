[![Python application](https://github.com/pybamm-team/liionpack/actions/workflows/python-app.yml/badge.svg)](https://github.com/pybamm-team/liionpack/actions/workflows/python-app.yml)
[![Documentation Status](https://readthedocs.org/projects/liionpack/badge/?version=latest)](https://liionpack.readthedocs.io/en/latest/?badge=latest)

# Overview of liionpack
*liionpack* takes a 1D PyBaMM model and makes it into a pack. You can either specify
the configuration e.g. 16 cells in parallel and 2 in series (16p2s) or load a
netlist

===============================================================================
Example Usage
===============================================================================

The following code block illustrates how to use liionpack to perform a simulation:

```
import liionpack as lp
netlist = lp.setup_circuit(Np=16, Ns=2, Rb=1e-4, Rc=1e-2, Ri=1e-3, V=4.0, I=80.0)
protocol = lp.generate_protocol()
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)
output = lp.solve(protocol=protocol)

```
