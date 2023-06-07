#
# Test Power Controlled Circuit Solve
#

import liionpack as lp
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')


# Define parameters
Np = 100
Ns = 1

# Generate netlist
netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1.5e-4, Rc=2e-2, Ri=5e-2, V=4.0, I=2.0)

power = 55.0

I_map = netlist["desc"].str.find("I") > -1
Terminal_Node = np.array(netlist[I_map].node1)[0]


V_node, I_batt = lp.solve_circuit_vectorized(netlist, power=power)
V_Terminal = V_node[Terminal_Node]
print("Terminal voltage [V]:", V_Terminal)
Current = -np.sum(I_batt)
print("Current [A]:", Current)
power_guess = V_Terminal * Current
print("Power [W]:", power_guess)
