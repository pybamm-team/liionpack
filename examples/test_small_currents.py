"""
A basic example of a pack simulation consisting of two 16 parallel cells
connected in series for a total of 32 cells.
"""

import liionpack as lp
import pybamm
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

lp.set_logging_level("NOTICE")

Np=16
Ns=2
I_app = Np * 2.0
# Generate the netlist
netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1e-4)



# Define a cycling experiment using PyBaMM
experiment = pybamm.Experiment([
    f"Charge at {I_app} A for 1 minutes",
    "Rest for 5 minutes",
    f"Discharge at {I_app} A for 1 minutes",
    "Rest for 5 minutes"],
    period="1 seconds")

# Define the PyBaMM parameters
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

inputs = {"Total heat transfer coefficient [W.m-2.K-1]" : np.ones(Np*Ns) * 10}

# Solve the pack
output = lp.solve(netlist=netlist,
                  sim_func=lp.thermal_simulation,
                  parameter_values=parameter_values,
                  experiment=experiment,
                  output_variables=None,
                  initial_soc=0.5,
                  inputs=inputs,
                  nproc=12)

# Plot the pack and individual cell results
lp.plot_pack(output)
lp.plot_cells(output)
lp.show_plots()
