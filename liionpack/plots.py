from lcapy import Circuit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sympy import init_printing
import textwrap

cmap = plt.cm.coolwarm
cmap = plt.cm.gist_rainbow
cmap = plt.cm.cool

init_printing(pretty_print=False)
lp_context = {
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


def draw_circuit(netlist, **kwargs):
    """
    Draw a latex version of netlist circuit
    N.B only works with generated netlists not imported ones.

    Args:
        netlist (pandas.DataFrame):
            A netlist of circuit elements with format. desc, node1, node2, value.

    Example:
        >>> import liionpack as lp
        >>> net = lp.setup_circuit(Np=3, Ns=1, Rb=1e-4, Rc=1e-2, Ri=5e-2, V=3.2, I=80.0)
        >>> lp.draw_circuit(net)
    """
    cct = Circuit()
    V_map = netlist["desc"].str.find("V") > -1
    I_map = netlist["desc"].str.find("I") > -1
    net2 = netlist.copy()
    net2.loc[V_map, ("node1")] = netlist["node2"][V_map]
    net2.loc[V_map, ("node2")] = netlist["node1"][V_map]
    net2.loc[I_map, ("node1")] = netlist["node2"][I_map]
    net2.loc[I_map, ("node2")] = netlist["node1"][I_map]

    for index, row in net2.iterrows():
        # print(row['desc'])
        string = ""
        direction = ""
        for ei, col in enumerate(row.iteritems()):
            if ei < 4:
                if col[0] == "desc":
                    if col[1][0] == "V":
                        direction = "up"
                    elif col[1][0] == "I":
                        direction = "up"
                    elif col[1][0] == "R":
                        if col[1][1] == "b":
                            if col[1][2] == "n":
                                direction = "right"
                            else:
                                direction = "left"
                        else:
                            direction = "down"
                string = string + str(col[1]) + " "

        string = string + "; " + direction
        cct.add(string)

    cct.draw(**kwargs)


def _text_color(vals, vmin, vmax, cmap):
    """
    Returns list of either black or white to write text, depending on whether
    plotted color is closer to white or black

    Args:
        vals (TYPE): DESCRIPTION.
        vmin (TYPE): DESCRIPTION.
        vmax (TYPE): DESCRIPTION.
        cmap (TYPE): DESCRIPTION.

    Returns:
        list: DESCRIPTION.

    """
    # return list of either black or white to write text, depending on whether
    # plotted color is closer to white or black
    cm = mpl.cm.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    val_cm = cm(norm(vals))[:, :3]
    val_norm = np.dot(val_cm, [0.2989, 0.5870, 0.1140])
    return ["k" if v > 0.5 else "w" for v in val_norm]


def _cell_text(ax, X, Y, vals, prec, text_colors):
    """

    Args:
        ax (TYPE): DESCRIPTION.
        vals (TYPE): DESCRIPTION.
        prec (TYPE): DESCRIPTION.
        text_colors (TYPE): DESCRIPTION.

    """
    # X_pos, Y_pos = cell_XY_positions()
    for i, val in enumerate(vals):
        ax.text(
            x=X[i],
            y=Y[i],
            s="{:.{}f}".format(val, prec),
            color=text_colors[i],
            ha="center",
            va="center",
        )


def _cell_text_numbers(ax, X, Y, text_colors):
    """

    Args:
        ax (TYPE): DESCRIPTION.
        text_colors (TYPE): DESCRIPTION.

    """
    # X_pos, Y_pos = cell_XY_positions()
    y_offset = 0.005
    for i, [x_pos, y_pos] in enumerate(zip(X, Y)):
        ax.text(
            x=x_pos,
            y=y_pos - y_offset,
            s="{:d}".format(i + 1),
            color=text_colors[i],
            ha="center",
            va="top",
            fontsize=7,
        )


def cell_scatter_plot(ax, X, Y, c, text_prec=1, **kwargs):
    """

    Args:
        ax (matplotlib.axes): axis to plot on.
        X (np.ndarray): x-coordinate of the battery
        Y (np.ndarray): y-coordinate of the battery
        c (like plt.scatter c kwarg): colors to plot scatter with.
        text_prec (int): precision to write text of values on cells.
        **kwargs : plt.scatter kwargs.

    """

    # X_pos, Y_pos = cell_XY_positions()

    # set size of markers
    diameter = 21.44 / 1000
    area = np.pi * (diameter / 2) ** 2
    s = area * 4 / np.pi  # points
    s = s * 72 / 0.0254 * 1000  # some scaling...

    # scatter plot
    sc = ax.scatter(X, Y, s, c, **kwargs)
    # set limits
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.06, 0.06)
    # colorbar
    plt.colorbar(sc, ax=ax, orientation="vertical")
    # set axis equal
    ax.set_aspect("equal")
    ax.set_axis_off()

    vmin, vmax = sc.get_clim()
    if "cmap" in kwargs:
        cmap = kwargs.get("cmap")
    else:
        cmap = "viridis"

    text_colors = _text_color(c, vmin, vmax, cmap)
    # write cell text
    _cell_text(ax, X, Y, c, text_prec, text_colors)
    _cell_text_numbers(ax, X, Y, text_colors)


def plot_pack(output):
    """
    Plot the battery pack voltage and current.

    Args:
        output (dict):
            Output from liionpack.solve which contains pack and cell variables.
    """

    # Get pack level results
    time = output["Time [s]"]
    v_pack = output["Pack terminal voltage [V]"]
    i_pack = output["Pack current [A]"]
    colors = cmap(np.linspace(0, 1, 2))
    with plt.rc_context(lp_context):
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


def plot_cells(output):
    """
    Plot results for the battery cells.

    Args:
        output (dict):
            Output from liionpack.solve which contains pack and cell variables.
    """

    # Get time and results for battery cells
    time = output["Time [s]"]
    cell_vars = [k for k in output.keys() if len(output[k].shape) > 1]

    # Get number of cells and setup colormap
    n = output[cell_vars[0]].shape[-1]
    colors = cmap(np.linspace(0, 1, n))

    # Create plot figures for cell variables
    with plt.rc_context(lp_context):
        for var in cell_vars:
            _, ax = plt.subplots(tight_layout=True)
            for i in range(n):
                ax.plot(time, output[var][:, i], color=colors[i])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(textwrap.fill(var, 45))
            ax.ticklabel_format(axis="y", scilimits=[-5, 5])


def plot_output(output):
    """
    Plot all results for pack and cells

    Args:
        output (dict):
            Output from liionpack.solve which contains pack and cell variables.

    """
    plot_pack(output)
    plot_cells(output)


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
    colors = cmap(np.linspace(0, 1, 4))
    with plt.rc_context(lp_context):
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

    with plt.rc_context(lp_context):
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
