#
# Paper example
#

import liionpack as lp
import pybamm

# Generate the netlist
netlist = lp.setup_circuit(Np=4, Ns=1, Rb=1e-3, Rc=1e-2)

# Define some additional variables to output
output_variables = [
    "X-averaged negative particle surface concentration [mol.m-3]",
    "X-averaged positive particle surface concentration [mol.m-3]",
]

# Cycling experiment, using PyBaMM
experiment = pybamm.Experiment(
    [
        "Charge at 5 A for 30 minutes",
        "Rest for 15 minutes",
        "Discharge at 5 A for 30 minutes",
        "Rest for 30 minutes",
    ],
    period="10 seconds",
)

# PyBaMM battery parameters
parameter_values = pybamm.ParameterValues("Chen2020")

# Solve the pack problem
output = lp.solve(
    netlist=netlist,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=0.5,
)

# Display the results
lp.plot_output(output, color="white")

# Draw the circuit at final state
lp.draw_circuit(netlist, cpt_size=1.0, node_spacing=2.2)
