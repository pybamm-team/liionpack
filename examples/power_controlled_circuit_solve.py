#
# Test Power Controlled Circuit Solve
#

import liionpack as lp
import numpy as np
import matplotlib.pyplot as plt
import pybamm

plt.close("all")


# Define parameters
Np = 1
Ns = 3

# Generate netlist
netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1.5e-4, Rc=2e-2, Ri=5e-2, V=4.0, I=2.2)

current = 2.0
power = 10.0


I_map = netlist["desc"].str.find("I") > -1
Terminal_Node = np.array(netlist[I_map].node1)[0]

V_node, I_batt, terminal_current, terminal_voltage, terminal_power = (
    lp.solve_circuit_vectorized(netlist, current=10.0, power=None)
)
V_Terminal = V_node[Terminal_Node]
print("Terminal voltage [V]:", terminal_voltage)
print("Current [A]:", terminal_current)
print("Power [W]:", terminal_power)

p = np.random.randint(5.0, 15.0, 100)
t = np.arange(len(p))

power_step = pybamm.experiment.step.power(50.0, duration="1000 s", period=10)
random_power_step = pybamm.experiment.step.power(np.vstack((t, p)).T)

# Cycling experiment
experiment = pybamm.Experiment(
    [power_step],
    period="1 s",
)
parameter_values = pybamm.ParameterValues("Chen2020")
output_variables = ["Terminal voltage [V]", "Current [A]", "Power [W]"]
# Solve pack
output = lp.solve(
    netlist=netlist,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=0.5,
)
lp.plot_output(output)
