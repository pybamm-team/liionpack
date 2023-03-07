"""
Example of saving the output from a 16p2s battery pack simulation. Examples
are given for CSV, NumPy `.npy`, and NumPy `.npz` file formats.
"""

import liionpack as lp
import pybamm
import numpy as np
import os

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
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)
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

# Save simulation output to CSV files
lp.save_to_csv(output)

# Save simulation output to Numpy npy files
lp.save_to_npy(output)

# Save simulation output to a compressed NumPy npz file
lp.save_to_npzcomp(output)
