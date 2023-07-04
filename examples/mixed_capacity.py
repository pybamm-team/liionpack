"""
Example of running a simulation with batteries of different size.
"""

import liionpack as lp
import pybamm
import numpy as np

lp.logger.setLevel("NOTICE")

# Define parameters
Np = 2
Ns = 3
Iapp = 10

# Generate the netlist and output variables
netlist = lp.setup_circuit(Np=Np,
                           Ns=Ns,
                           Rb=1.5e-3,
                           Rc=1e-2,
                           Ri=5e-2,
                           V=4.0,
                           I=Iapp,
                           configuration="series-groups")

lp.draw_circuit(netlist)

# Cycling experiment
experiment = pybamm.Experiment(
    [
        f"Discharge at {Iapp} A for 30 minutes",
        "Rest for 30 minutes",
    ],
    period="10 seconds",
)

# PyBaMM parameters
param = pybamm.ParameterValues("Chen2020")

w_original = param["Electrode width [m]"]

param.update(
    {
        "Electrode width [m]": "[input]",
    }
)

new_widths = np.ones(Np * Ns) * w_original

# Divide the capacity by 2 for 1 half of the batteries
new_widths[:3] = w_original / 2

inputs = {
    "Electrode width [m]": new_widths,
}

# Solve pack
output = lp.solve(
    netlist=netlist,
    parameter_values=param,
    experiment=experiment,
    initial_soc=None,
    inputs=inputs,
)

# Plot results
lp.plot_output(output)
lp.show_plots()
