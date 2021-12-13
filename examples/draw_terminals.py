#
# Plot a netlist
#

import liionpack as lp
import matplotlib.pyplot as plt

plt.close("all")

left = lp.setup_circuit(Np=3, Ns=1, terminals="left")
lp.draw_circuit(left)

right = lp.setup_circuit(Np=3, Ns=1, terminals="right")
lp.draw_circuit(right)

left_right = lp.setup_circuit(Np=3, Ns=1, terminals="left-right")
lp.draw_circuit(left_right)

right_left = lp.setup_circuit(Np=3, Ns=1, terminals="right-left")
lp.draw_circuit(right_left)

middle = lp.setup_circuit(Np=3, Ns=1, terminals=[1, 1])
lp.draw_circuit(middle)
