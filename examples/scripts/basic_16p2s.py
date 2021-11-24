"""
A basic example of a pack simulation consisting of two 16 parallel cells
connected in series for a total of 32 cells.
"""

import liionpack as lp
import pybamm
lp.set_logging_level("NOTICE")

# Generate the netlist
netlist = lp.setup_circuit(Np=16, Ns=2)

# Define additional output variables
output_variables = [
    'X-averaged negative particle surface concentration [mol.m-3]',
    'X-averaged positive particle surface concentration [mol.m-3]']

# Define a cycling experiment using PyBaMM
experiment = pybamm.Experiment([
    "Charge at 20 A for 30 minutes",
    "Rest for 15 minutes",
    "Discharge at 20 A for 30 minutes",
    "Rest for 30 minutes"],
    period="10 seconds")

# Define the PyBaMM parameters
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

# Solve the pack
output = lp.solve(netlist=netlist,
                  parameter_values=parameter_values,
                  experiment=experiment,
                  output_variables=output_variables)

# Plot the pack and individual cell results
lp.plot_pack(output)
lp.plot_cells(output)
lp.show_plots()
