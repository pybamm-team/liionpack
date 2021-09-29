# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:27:40 2021

@author: Tom
"""


import liionpack as lp
import numpy as np
import pybamm
import matplotlib.pyplot as plt

plt.close('all')
pybamm.logger.setLevel('NOTICE')

# Generate the netlist
netlist = lp.setup_circuit(Np=4, Ns=1, Rb=1.5e-3, Rc=1e-2, Ri=5e-2, V=4.0, I=5.0)
lp.draw_circuit(netlist, scale_factor=0.5, cpt_size=1.0, dpi=150, node_spacing=2.5)
output_variables = [  
    'X-averaged total heating [W.m-3]',
    'Volume-averaged cell temperature [K]',
    'X-averaged negative particle surface concentration [mol.m-3]',
    'X-averaged positive particle surface concentration [mol.m-3]',
    ]
# Heat transfer coefficients
htc = np.ones(4) * 10
# Cycling protocol
protocol = lp.generate_protocol(I_chg=5, I_dch=-5, chg_first=True)
# PyBaMM parameters
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)
# Solve pack
output = lp.solve(netlist=netlist,
                  parameter_values=parameter_values,
                  protocol=protocol,
                  output_variables=output_variables,
                  htc=htc)