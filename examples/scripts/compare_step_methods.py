# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:13:37 2021

@author: tom
"""

import liionpack as lp
import numpy as np
import pybamm
import matplotlib.pyplot as plt

plt.close("all")
pybamm.logger.setLevel("NOTICE")

# Generate the netlist
netl1 = lp.setup_circuit(Np=4, Ns=1, Rb=1.5e-3, Rc=1e-2, Ri=5e-2, V=4.0, I=5.0)
netl2 = lp.setup_circuit(Np=4, Ns=1, Rb=1.5e-3, Rc=1e-2, Ri=5e-2, V=4.0, I=5.0)
output_variables = None

# Heat transfer coefficients
htc = np.ones(4) * 10

# Cycling experiment
experiment = pybamm.Experiment(
    [
        "Discharge at 5 A for 30 minutes",
    ],
    period="10 seconds",
)

# PyBaMM parameters
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

# Solve pack
# Serial step
srlout = lp.solve(
    netlist=netl1,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    htc=htc,
    mapped=False,
)
# Casadi mapped step
mapout = lp.solve(
    netlist=netl2,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    htc=htc,
    mapped=True,
)

print(np.allclose(srlout["Cell current [A]"], mapout["Cell current [A]"]))
