#
# Plot a netlist
#

import liionpack as lp
import matplotlib.pyplot as plt
plt.close('all')

netlist = lp.setup_circuit(Np=5, Ns=1, terminals=[0, 4])
lp.simple_netlist_plot(netlist)
lp.draw_circuit(netlist, cpt_size=1.0, dpi=300, node_spacing=2.5)
