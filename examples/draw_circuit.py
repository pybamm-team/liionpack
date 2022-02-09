"""
Set up a simulation and draw the circuit.
"""

import liionpack as lp
import pybamm

lp.logger.setLevel("NOTICE")

# Define parameters
Np = 4
Ns = 1

# Generate the netlist and draw circuit
netlist = lp.setup_circuit(Np=Np, Ns=Ns, I=5)
lp.draw_circuit(netlist, node_spacing=2.5)

output_variables = [
    "X-averaged negative particle surface concentration [mol.m-3]",
    "X-averaged positive particle surface concentration [mol.m-3]",
]

# Cycling experiment
experiment = pybamm.Experiment(
    [
        "Charge at 5 A for 30 minutes",
        "Rest for 15 minutes",
        "Discharge at 5 A for 30 minutes",
        "Rest for 30 minutes",
    ],
    period="10 seconds",
)

# PyBaMM parameters
parameter_values = pybamm.ParameterValues("Chen2020")

# Solve pack
output = lp.solve(
    netlist=netlist,
    parameter_values=parameter_values,
    experiment=experiment,
    initial_soc=0.5,
    output_variables=output_variables,
)

lp.plot_output(output)
lp.show_plots()
