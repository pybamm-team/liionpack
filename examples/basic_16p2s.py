"""
A basic example of a pack simulation consisting of two sets of 16 parallel
cells connected in series for a total of 32 cells.
"""

import liionpack as lp
import pybamm
import numpy as np
import os

lp.set_logging_level("NOTICE")

# Define parameters
Np = 16
Ns = 2
Iapp = 20

# Generate the netlist
netlist = lp.setup_circuit(Np=Np, Ns=Ns)

# Define additional output variables
output_variables = ["Volume-averaged cell temperature [K]"]

# Define a cycling experiment using PyBaMM
experiment = pybamm.Experiment(
    [
        f"Charge at {Iapp} A for 30 minutes",
        "Rest for 15 minutes",
        f"Discharge at {Iapp} A for 30 minutes",
        "Rest for 30 minutes",
    ],
    period="10 seconds",
)

# Define the PyBaMM parameters
parameter_values = pybamm.ParameterValues("Chen2020")
inputs = {"Total heat transfer coefficient [W.m-2.K-1]": np.ones(Np * Ns) * 10}

# Solve the pack
output = lp.solve(
    netlist=netlist,
    sim_func=lp.thermal_simulation,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=0.5,
    inputs=inputs,
    nproc=os.cpu_count(),
    manager="casadi",
)

# Plot the pack and individual cell results
lp.plot_pack(output)
lp.plot_cells(output)
lp.show_plots()
