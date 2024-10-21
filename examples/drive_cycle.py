"""
Example of using US06 drive cycle data for a battery pack simulation.
"""

import pybamm
import os
import pandas as pd
import liionpack as lp

os.chdir(pybamm.__path__[0] + "/..")

# Define parameters
Np = 4
Ns = 1

# Generate netlist
netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1.5e-3, Rc=1e-2, Ri=5e-2, V=4.0, I=5.0)

# Define the PyBaMM parameters
parameter_values = pybamm.ParameterValues("Chen2020")

# Import drive cycle from file
drive_cycle = pd.read_csv(
    pybamm.DataLoader().get_data("US06.csv"), comment="#", header=None
).to_numpy()

experiment = pybamm.Experiment([pybamm.step.current(drive_cycle)])

# Solve pack
output = lp.solve(
    netlist=netlist,
    parameter_values=parameter_values,
    experiment=experiment,
    initial_soc=0.5,
)

# Plot results
lp.plot_output(output)
lp.show_plots()
