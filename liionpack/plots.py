#
# Postprocessing plot functions
#

import liionpack as lp
import numpy as np
import matplotlib.pyplot as plt
from sympy import init_printing
import textwrap

init_printing(pretty_print=False)


def lp_cmap(color="dark"):
    """
    Return the colormap to use in plots

    Args:
        color (string):
            The color-scheme for plotting, default="dark".

    Returns:
        cmap (matplotlib.cm):
            The colormap for matplotlib to plot

    """
    if color == "dark":
        return plt.cm.cool
    else:
        return plt.cm.coolwarm


def lp_context(color="dark"):
    """
    Return the liionpack matplotlib rc_context for plotting

    Args:
        color (string):
            The color-scheme for plotting, default="dark"

    Returns:
        context (dict):
            The options to pass to matplotlib.pyplot.rc_context

    """
    if color == "dark":
        context = {
            "text.color": "white",
            "axes.edgecolor": "white",
            "axes.titlecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "figure.facecolor": "#323232",
            "axes.facecolor": "#323232",
            "axes.grid": False,
            "axes.labelsize": "large",
            "figure.figsize": (8, 6),
        }
    else:
        context = {
            "text.color": "black",
            "axes.edgecolor": "black",
            "axes.titlecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.labelsize": "large",
            "figure.figsize": (8, 6),
        }
    return context


def draw_circuit(
    netlist,
    cpt_size=1.0,
    dpi=300,
    node_spacing=2.0,
    scale=1.0,
    help_lines=0.0,
    font="\scriptsize",
    label_ids=True,
    label_values=True,
    draw_nodes=True,
    label_nodes="primary",
    style="american",
):
    """
    Draw a latex version of netlist circuit
    N.B only works with generated netlists not imported ones.

    Args:
        netlist (pandas.DataFrame):
            A netlist of circuit elements with format. desc, node1, node2, value.
        cpt_size (float):
            component size, default 1.0
        dpi (int):
            dots per inch, default 300
        node_spacing (float):
            spacing between component nodes, default 2.0
        scale (float):
            schematic scale factor, default 1.0
        help_lines (float):
            distance between lines in grid, default 0.0 (disabled)
        font (string):
            LaTex font size, default \scriptsize
        label_ids (bool):
            Show component ids, default True
        label_values (bool):
            Display component values, default True
        draw_nodes (bool):
            True to show all nodes (default), False to show no nodes,'primary' to show
            primary nodes, 'connections' to show nodes that connect more than
            two components, 'all' to show all nodes.
        label_nodes (bool):
            True to label all nodes, False to label no nodes, 'primary' to label
            primary nodes (default), 'alpha' to label nodes starting with a letter,
            'pins' to label nodes that are pins on a chip, 'all' to label all nodes
        style (string):
            'american', 'british', or 'european'

    Example:
        >>> import liionpack as lp
        >>> net = lp.setup_circuit(Np=3, Ns=1, Rb=1e-4, Rc=1e-2, Ri=5e-2, V=3.2, I=80.0)
        >>> lp.draw_circuit(net)
    """
    cct = lp.make_lcapy_circuit(netlist)
    kwargs = {
        "cpt_size": cpt_size,
        "dpi": dpi,
        "node_spacing": node_spacing,
        "scale": scale,
        "help_lines": help_lines,
        "font": font,
        "label_ids": label_ids,
        "label_values": label_values,
        "draw_nodes": draw_nodes,
        "label_nodes": label_nodes,
        "style": style,
    }
    cct.draw(**kwargs)


def plot_pack(output, color="dark"):
    """
    Plot the battery pack voltage and current.

    Args:
        output (dict):
            Output from liionpack.solve which contains pack and cell variables.
        color (string):
            The color-scheme for plotting, default="dark"
    """

    # Get pack level results
    time = output["Time [s]"]
    v_pack = output["Pack terminal voltage [V]"]
    i_pack = output["Pack current [A]"]

    context = lp_context(color)
    cmap = lp_cmap(context)

    colors = cmap(np.linspace(0, 1, 2))
    with plt.rc_context(context):
        # Plot pack voltage and current
        _, ax = plt.subplots(tight_layout=True)
        ax.plot(time, v_pack, color=colors[0], label="simulation")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pack terminal voltage [V]", color=colors[0])
        ax.grid(False)
        ax2 = ax.twinx()
        ax2.plot(time, i_pack, color=colors[1], label="simulation")
        ax2.set_ylabel("Pack current [A]", color=colors[1])
        ax2.set_title("Pack Summary")


def plot_cells(output, color="dark"):
    """
    Plot results for the battery cells.

    Args:
        output (dict):
            Output from liionpack.solve which contains pack and cell variables.
        color (string):
            The color-scheme for plotting, default="dark"
    """

    # Get time and results for battery cells
    time = output["Time [s]"]
    cell_vars = [k for k in output.keys() if len(output[k].shape) > 1]

    context = lp_context(color)
    cmap = lp_cmap(context)

    # Get number of cells and setup colormap
    n = output[cell_vars[0]].shape[-1]
    colors = cmap(np.linspace(0, 1, n))

    # Create plot figures for cell variables
    with plt.rc_context(context):
        for var in cell_vars:
            _, ax = plt.subplots(tight_layout=True)
            for i in range(n):
                ax.plot(time, output[var][:, i], color=colors[i])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(textwrap.fill(var, 45))
            ax.ticklabel_format(axis="y", scilimits=[-5, 5])


def plot_output(output, color="dark"):
    """
    Plot all results for pack and cells

    Args:
        output (dict):
            Output from liionpack.solve which contains pack and cell variables.
        color (string):
            The color-scheme for plotting, default="dark"

    """
    plot_pack(output, color)
    plot_cells(output, color)


def show_plots():  # pragma: no cover
    """
    Wrapper function for the Matplotlib show() function.
    """
    plt.show()


def simple_netlist_plot(netlist):
    """
    Simple matplotlib netlist plot with colored lines for different elements

    Args:
        netlist (TYPE): DESCRIPTION.

    """
    plt.figure()
    for row in netlist.iterrows():
        elem, node1, node2, value, x1, y1, x2, y2 = row[1]
        if elem[0] == "I":
            color = "g"
        elif elem[:2] == "Rs":
            color = "r"
        elif elem[:2] == "Rb":
            color = "k"
        elif elem[:2] == "Ri":
            color = "y"
        elif elem[:2] == "Rt":
            color = "pink"
        elif elem[0] == "V":
            color = "b"
        else:
            color = "k"
        plt.scatter([x1, x2], [y1, y2], c="k")
        plt.plot([x1, x2], [y1, y2], c=color)


def compare_solution_output(a, b):
    r"""
    Compare two solutions Terminal Voltage [V] and Current [A]. Solutions can
    be PyBaMM.Solution or dict output from Liionpack solve.

    Args:
        a (dict / PyBaMM.Solution):
            Output from solve.
        b (dict / PyBaMM.Solution):
            Output from solve.

    """
    # Get pack level results
    if a.__class__ is dict:
        time_a = a["Time [s]"]
        v_a = a["Pack terminal voltage [V]"]
        i_a = a["Pack current [A]"]
        title_a = "a) Liionpack Simulation"
    else:
        time_a = a["Time [s]"].entries
        v_a = a["Terminal voltage [V]"].entries
        i_a = a["Current [A]"].entries
        title_a = "a) PyBaMM Simulation"
    if b.__class__ is dict:
        time_b = b["Time [s]"]
        v_b = b["Pack terminal voltage [V]"]
        i_b = b["Pack current [A]"]
        title_b = "b) Liionpack Simulation"
    else:
        time_b = b["Time [s]"].entries
        v_b = b["Terminal voltage [V]"].entries
        i_b = b["Current [A]"].entries
        title_b = "b) PyBaMM Simulation"
    cmap = lp_cmap()
    colors = cmap(np.linspace(0, 1, 4))
    with plt.rc_context(lp_context()):
        # Plot pack voltage and current
        _, (axl, axr) = plt.subplots(
            1, 2, tight_layout=True, figsize=(15, 10), sharex=True, sharey=True
        )
        axl.plot(time_a, v_a, color=colors[0], label="simulation")
        axl.set_xlabel("Time [s]")
        axl.set_ylabel("Terminal voltage [V]", color=colors[0])
        axl2 = axl.twinx()
        axl2.plot(time_a, i_a, color=colors[1], label="simulation")
        axl2.set_ylabel("Current [A]", color=colors[1])
        axl2.set_title(title_a)
        axr.plot(time_b, v_b, color=colors[2], label="simulation")
        axr.set_xlabel("Time [s]")
        axr.set_ylabel("Terminal voltage [V]", color=colors[2])
        axr2 = axr.twinx()
        axr2.plot(time_b, i_b, color=colors[3], label="simulation")
        axr2.set_ylabel("Current [A]", color=colors[3])
        axr2.set_title(title_b)


def plot_cell_data_image(netlist, data, tick_labels=True, figsize=(8, 6)):
    r"""
    Plot the cell data for all cells at a particular point in time in an image
    format using the node coordinates in the netlist to arrange the cells.

    Args:
        netlist (pandas.DataFrame):
            A netlist of circuit elements with format desc, node1, node2, value.
        data (numpy.array):
            The data to be plotted for each cell.
        tick_labels boolean:
            Show the Np and Ns cell indices.
        figsize (tuple):
            The figzise in inches.

    """
    V_map = netlist["desc"].str.find("V") > -1
    vlist = netlist[V_map]
    n1x = np.unique(vlist["node1_x"])
    n1y = np.unique(vlist["node1_y"])
    Nx = len(n1x)
    Ny = len(n1y)
    for ix in range(Nx):
        vlist.loc[vlist["node1_x"] == n1x[ix], ("node1_x")] = ix
    for iy in range(Ny):
        vlist.loc[vlist["node1_y"] == n1y[iy], ("node1_y")] = iy

    im = np.ones([Nx, Ny])
    im[np.array(vlist["node1_x"]), np.array(vlist["node1_y"])] = data

    cmap = lp_cmap()
    with plt.rc_context(lp_context()):
        fig, ax = plt.subplots(figsize=figsize)
        mappable = ax.imshow(im.T, cmap=cmap)
        # Major ticks
        ax.set_xticks(np.arange(0, Nx, 1))
        ax.set_yticks(np.arange(0, Ny, 1))
        if tick_labels:
            # Labels for major ticks
            ax.set_xticklabels(np.arange(0, Nx, 1))
            ax.set_yticklabels(np.arange(0, Ny, 1))
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        # Minor ticks
        ax.set_xticks(np.arange(-0.5, Nx, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, Ny, 1), minor=True)
        # Gridlines based on minor ticks
        ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
        plt.colorbar(mappable)
        plt.tight_layout()
