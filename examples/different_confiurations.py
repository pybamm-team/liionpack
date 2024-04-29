import liionpack as lp
import pybamm
import numpy as np
import os
import matplotlib.pyplot as plt

plt.close('all')
lp.set_logging_level("NOTICE")

names = ["8P-1S", "4P-2S", "2S-4P", "2P-4S", "4S-2P", "8S-1P"]
case = 0
C = 5.0
# Define parameters

_Np = [8, 4, 4, 2, 2, 1]
_Ns = [1, 2, 2, 4, 4, 8]
_I = C*np.array(_Np)
configurations = [
    "parallel-strings",
    "parallel-strings",
    "series-groups",
    "parallel-strings",
    "series-groups",
    "parallel-strings"
    ]

Iapp = _I[case]
Np = _Np[case]
Ns = _Ns[case]
Nb = Np * Ns
configuration = configurations[case]
name = names[case]

# Generate the netlist
netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1e-4, I=C, configuration=configuration)
# lp.draw_circuit(netlist)

# Define additional output variables
output_variables = ["Volume-averaged cell temperature [K]"]

# Define a cycling experiment using PyBaMM
experiment = pybamm.Experiment(
    [
        f"Discharge at {Iapp} A for 1 hour or until 3.0 V",
        "Rest for 30 minutes",
        f"Charge at {Iapp / 2} A for 2 hours or until 4.1 V",
        "Rest for 30 minutes",
    ],
    period="10 seconds",
)

# Define the PyBaMM parameters
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update({
    "Negative electrode active material volume fraction": "[input]",
    "Positive electrode active material volume fraction": "[input]",
    })
inputs = {
    "Total heat transfer coefficient [W.m-2.K-1]": 10 + (np.random.random(Nb) * 1.0),
    "Negative electrode active material volume fraction": 0.75 + (np.random.random(Nb) * 0.01),
    "Positive electrode active material volume fraction": 0.66 + (np.random.random(Nb) * 0.01),
    }

# Solve the pack
output = lp.solve(
    netlist=netlist,
    sim_func=lp.thermal_simulation,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=None,
    inputs=inputs,
    nproc=os.cpu_count(),
    manager="casadi",
)

# Plot the pack and individual cell results
lp.plot_pack(output)
lp.plot_cells(output)

lp.show_plots()


