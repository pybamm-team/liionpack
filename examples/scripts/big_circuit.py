# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:27:40 2021

@author: Tom
"""

import liionpack as lp
import numpy as np
import pybamm
import matplotlib.pyplot as plt

plt.close("all")
pybamm.logger.setLevel("NOTICE")

Np = 16
Ns = 96
Nspm = Np * Ns
I_app = Np * 2.0

# Generate the netlist
netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1e-4, Rc=1e-2, Ri=5e-2, V=3.2, I=I_app)

output_variables = [
    "X-averaged total heating [W.m-3]",
    "Volume-averaged cell temperature [K]",
    "X-averaged negative particle surface concentration [mol.m-3]",
    "X-averaged positive particle surface concentration [mol.m-3]",
]

# Heat transfer coefficients
htc = np.ones(Nspm) * 10

# Cycling experiment
experiment = pybamm.Experiment(
    [
        f"Discharge at {I_app} A for 1000 seconds",
        "Rest for 1000 seconds",
        f"Charge at {I_app} A for 1000 seconds",
        "Rest for 1000 seconds",
    ],
    period="10 seconds",
)

# PyBaMM parameters
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

# lp.create_init_funcs(parameter_values)

# Solve pack
output = lp.solve(
    netlist=netlist,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    htc=htc,
)

lp.plot_output(output)
