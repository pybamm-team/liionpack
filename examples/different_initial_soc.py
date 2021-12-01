#
# Simulation with two batteries of different initial soc
#

import liionpack as lp
import pybamm
import matplotlib.pyplot as plt

plt.close("all")
lp.logger.setLevel("NOTICE")

# Generate the netlist
netlist = lp.setup_circuit(Np=2, Ns=1, Rb=1.5e-3, Rc=1e-2, Ri=5e-2, V=4.0, I=5.0)
output_variables = [
    "X-averaged negative particle surface concentration [mol.m-3]",
    "X-averaged positive particle surface concentration [mol.m-3]",
]

I_app = 2.0
# Cycling experiment
experiment = pybamm.Experiment(
    [
        f"Charge at {I_app} A for 30 minutes",
        "Rest for 15 minutes",
        f"Discharge at {I_app} A for 30 minutes",
        "Rest for 30 minutes",
    ],
    period="10 seconds",
)

# PyBaMM parameters
chemistry = pybamm.parameter_sets.Chen2020
param = pybamm.ParameterValues(chemistry=chemistry)

c_s_n_init, c_s_p_init = lp.update_init_conc(param, SoC=[0.5, 0.51], update=False)

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

lp.plot_output(output)
