"""
Example of running a simulation with two batteries of different initial SOC.
"""

import liionpack as lp
import pybamm
import numpy as np

lp.logger.setLevel("NOTICE")

# Define parameters
Np = 2
Ns = 1
Iapp = 5

# Generate the netlist and output variables
netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1.5e-3, Rc=1e-2, Ri=5e-2, V=4.0, I=Iapp)
output_variables = [
    "X-averaged negative particle surface concentration [mol.m-3]",
    "X-averaged positive particle surface concentration [mol.m-3]",
]

# Cycling experiment
experiment = pybamm.Experiment(
    [
        f"Charge at {Iapp} A for 30 minutes",
        "Rest for 15 minutes",
        f"Discharge at {Iapp} A for 30 minutes",
        "Rest for 30 minutes",
    ],
    period="10 seconds",
)

# PyBaMM parameters
param = pybamm.ParameterValues("Chen2020")

c_s_n_init, c_s_p_init = lp.update_init_conc(
    param, SoC=np.array([0.5, 0.6]), update=False
)

param.update(
    {
        "Initial concentration in negative electrode [mol.m-3]": "[input]",
        "Initial concentration in positive electrode [mol.m-3]": "[input]",
    }
)

inputs = {
    "Initial concentration in negative electrode [mol.m-3]": c_s_n_init,
    "Initial concentration in positive electrode [mol.m-3]": c_s_p_init,
}

# Solve pack
output = lp.solve(
    netlist=netlist,
    parameter_values=param,
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=None,
    inputs=inputs,
)

# Plot results
lp.plot_output(output)
lp.show_plots()
