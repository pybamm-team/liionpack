import pybamm
import os
import pandas as pd
import liionpack as lp
import numpy as np

os.chdir(pybamm.__path__[0]+'/..')

netlist = lp.setup_circuit(Np=4, Ns=1, Rb=1.5e-3, Rc=1e-2, Ri=5e-2, V=4.0, I=5.0)

chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

# Heat transfer coefficients
htc = np.ones(4) * 10

# import drive cycle from file
drive_cycle = pd.read_csv("pybamm/input/drive_cycles/US06.csv",
                          comment="#", header=None).to_numpy()

experiment = pybamm.Experiment(operating_conditions=['Run US06 (A)'],
                               drive_cycles={'US06':  drive_cycle})

# PyBaMM parameters
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

# Solve pack
output = lp.solve(
    netlist=netlist,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=None,
    htc=htc,
)

lp.plot_output(output)