#
# Plot a netlist
#

import liionpack as lp
import matplotlib.pyplot as plt

plt.close("all")

netlist = lp.setup_circuit(Np=5, Ns=1, terminals=[2, 3])
# lp.simple_netlist_plot(netlist)
lp.draw_circuit(netlist)
