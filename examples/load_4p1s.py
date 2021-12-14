#
# Load a netlist to use in a liionpack simulation and compare results to a
# generated one
#

import liionpack as lp
import numpy as np


Rb = 1e-3
Ri = 1e-2
Rc = 1e-2
Rt = 1e-5
Rt2 = 1e-5
I = -2
V = 4.0

net1 = lp.read_netlist(filepath="4p1s.cir", Rb=Rb, Ri=Ri, Rc=Rc, Rt=Rt, I=I, V=V)
V1, I1 = lp.solve_circuit(net1)

net2 = lp.setup_circuit(Np=4, Ns=1, Rb=Rb, Ri=Ri, Rc=Rc, Rt=Rt2, I=I, V=V)
V2, I2 = lp.solve_circuit(net2)

# Nodal order is different
print("V match: ", np.allclose(np.sort(V1), np.sort(V2)))
print("I match: ", np.allclose(np.sort(I1), np.sort(I2)))

lp.draw_circuit(net2, cpt_size=1.0, dpi=300, node_spacing=2.5)
