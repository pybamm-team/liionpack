# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:27:40 2021

@author: Tom
"""


import liionpack as lp
import matplotlib.pyplot as plt
import numpy as np
import pybamm

plt.close('all')
# Circuit parameters
Np=16
Ns=2
Nspm = Np * Ns
R_bus=1e-4
R_series=1e-2
R_int=5e-2
I_app=80.0
ref_voltage = 3.2
# Generate the netlist
netlist = lp.setup_circuit(Np, Ns, Rb=R_bus, Rc=R_series, Ri=R_int, V=ref_voltage, I=I_app)
output_variables = [  
    'X-averaged total heating [W.m-3]',
    'Volume-averaged cell temperature [K]',
    'X-averaged negative particle surface concentration [mol.m-3]',
    'X-averaged positive particle surface concentration [mol.m-3]',
    ]
# Heat transfer coefficients
htc = np.ones(Nspm) * 10
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