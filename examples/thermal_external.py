#
# External thermal example
#

import liionpack as lp
import pybamm
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")


# Generate the netlist
netlist = lp.setup_circuit(Np=4, Ns=1, Rb=1e-3, Rc=1e-2)

# Define some additional variables to output
output_variables = [
    "Volume-averaged cell temperature [K]",
    "Volume-averaged total heating [W.m-3]",
]

# Cycling experiment, using PyBaMM
experiment = pybamm.Experiment(["Discharge at 5 A for 5 minutes"], period="10 seconds")

# PyBaMM battery parameters
parameter_values = pybamm.ParameterValues("Chen2020")

# Solve the pack problem
temps = np.ones(4) * 300 + np.arange(4) * 10
inputs = {"Input temperature [K]": temps}
output = lp.solve(
    netlist=netlist,
    sim_func=lp.thermal_external,
    inputs=inputs,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=0.5,
)

# Display the results
lp.plot_output(output, color="white")
