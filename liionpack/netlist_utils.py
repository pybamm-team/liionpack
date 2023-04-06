#
# Utility functions for loading and creating and solving circuits defined by
# netlists
#


import numpy as np
import codecs
import pandas as pd
import liionpack as lp
import os
import pybamm
import scipy as sp
from lcapy import Circuit


def read_netlist(
    filepath,
    Ri=None,
    Rc=None,
    Rb=None,
    Rt=None,
    I=None,
    V=None,
):
    """
    Assumes netlist has been saved by LTSpice with format Descriptor Node1 Node2 Value
    Any lines starting with * are comments and . are commands so ignore them
    Nodes begin with N so remove that
    Open ended components are not allowed and their nodes start with NC (no-connection)

    Args:
        filepath (str): Path to netlist circuit file '.cir' or '.txt'.
        Ri (float): Internal resistance ($\Omega$).
        Rc (float): Connection resistance ($\Omega$).
        Rb (float): Busbar resistance ($\Omega$).
        Rt (float): Terminal connection resistance ($\Omega$).
        I (float): Current (A).
        V (float): Initial battery voltage (V).

    Returns:
        pandas.DataFrame:
            A netlist of circuit elements with format desc, node1, node2, value.
    """

    # Read in the netlist
    if "." not in filepath:
        filepath += ".cir"
    if not os.path.isfile(filepath):
        temp = os.path.join(lp.CIRCUIT_DIR, filepath)
        if not os.path.isfile(temp):
            pass
        else:
            filepath = temp
    if ".cir" in filepath:
        with codecs.open(filepath, "r", "utf-16LE") as fd:
            Lines = fd.readlines()
    elif ".txt" in filepath:
        with open(filepath, "r") as f:
            Lines = f.readlines()
    else:
        raise FileNotFoundError(
            'Please supply a valid file with extension ".cir" or ".txt"'
        )
    # Ignore lines starting with * or .
    Lines = [l.strip("\n").split(" ") for l in Lines if l[0] not in ["*", "."]]
    Lines = np.array(Lines, dtype="<U16")

    # Read descriptions and nodes, strip N from nodes
    # Lines is desc | node1 | node2
    desc = Lines[:, 0]
    node1 = Lines[:, 1]
    node2 = Lines[:, 2]
    value = Lines[:, 3]
    try:
        value = value.astype(float)
    except ValueError:
        pass
    node1 = np.array([x.strip("N") for x in node1], dtype=int)
    node2 = np.array([x.strip("N") for x in node2], dtype=int)
    netlist = pd.DataFrame(
        {"desc": desc, "node1": node1, "node2": node2, "value": value}
    )

    # Populate the values based on the descriptions (element types)
    for name, val in [
        ("Ri", Ri),
        ("Rc", Rc),
        ("Rb", Rb),
        ("Rl", Rb),
        ("Rt", Rt),
        ("I", I),
        ("V", V),
    ]:
        if val is not None:
            # netlist["desc"] consists of entries like 'Ri13'
            # this map finds all the entries that start with (e.g.) 'Ri'
            name_map = netlist["desc"].str.find(name) > -1
            # then allocates the value to the corresponding indices
            netlist.loc[name_map, ("value")] = val

    lp.logger.notice("netlist " + filepath + " loaded")
    return netlist


def setup_circuit(
    Np=1,
    Ns=1,
    Ri=1e-2,
    Rc=1e-2,
    Rb=1e-4,
    Rt=1e-5,
    I=80.0,
    V=4.2,
    plot=False,
    terminals="left",
    configuration="parallel-strings",
):
    """
    Define a netlist from a number of batteries in parallel and series

    Args:
        Np (int): Number of batteries in parallel.
        Ns (int): Number of batteries in series.
        Ri (float): Internal resistance ($\Omega$).
        Rc (float): Connection resistance ($\Omega$).
        Rb (float): Busbar resistance ($\Omega$).
        Rt (float): Terminal connection resistance ($\Omega$).
        I (float): Current (A).
        V (float): Initial battery voltage (V).
        plot (bool): Plot the circuit.
        terminals (string): The location of the terminals. Can be "left", "right",
            "left-right", "right-left" or a list or array of node integers.
        configuration (string): The pack circuit configuration to use. Can be
            "parallel-strings" (default) or "series-groups"

    Returns:
        pandas.DataFrame:
            A netlist of circuit elements with format desc, node1, node2, value.

    """
    Nc = Np
    Nr = Ns * 3 + 1

    grid = np.arange(Nc * Nr).reshape([Nr, Nc])
    coords = np.indices(grid.shape)
    y = coords[0, :, :]
    x = coords[1, :, :]
    # make contiguous now instead of later when netlist is done as very slow
    mask = np.ones([Nr, Nc], dtype=bool)
    # This is no longer needed as terminals connect directly to battery
    # Guess could also add a terminal connection resistor though
    # mask[1:-1, 0] = False
    grid[mask] = np.arange(np.sum(mask)) + 1
    x = x[mask].flatten()
    y = y[mask].flatten()
    grid[~mask] = -2  # These should never be used

    # grid is a Nr x Nc matrix
    # 1st column is terminals only
    # 1st and last rows are busbars
    # Other rows alternate between series resistor and voltage source
    # For example if Np=1 and Nc=2,
    # grid = array([[ 0,  1], # busbar
    #                         # Rs
    #               [ 2,  3],
    #                         # V
    #               [ 4,  5],
    #                         # Ri
    #               [ 6,  7],
    #                         # Rs
    #               [ 8,  9],
    #                         # V
    #               [10, 11],
    #                         # Ri
    #               [12, 13]] # busbar)
    # Connections are across busbars in first and last rows, and down each column
    # See "01 Getting Started.ipynb"

    # Build data  with ['element type', node1, node2, value]
    netlist = []

    num_Rb = 0
    num_V = 0

    desc = []
    node1 = []
    node2 = []
    value = []

    # -ve busbars (bottom row of the grid)
    bus_nodes = [grid[0, :]]
    for nodes in bus_nodes:
        for i in range(len(nodes) - 1):
            # netline = []
            desc.append("Rbn" + str(num_Rb))
            num_Rb += 1
            node1.append(nodes[i])
            node2.append(nodes[i + 1])
            value.append(Rb)
    num_Rs = 0
    num_Ri = 0
    # Series resistors and voltage sources
    cols = np.arange(Nc)
    rows = np.arange(Nr)[:-1]
    rtype = ["Rc", "V", "Ri"] * Ns
    for col in cols:
        # Go down the column alternating Rs, V, Ri connections between nodes
        nodes = grid[:, col]
        for row in rows:
            if rtype[row] == "Rc":
                # Inter(c)onnection / weld
                desc.append(rtype[row] + str(num_Rs))
                num_Rs += 1
                val = Rc
            elif rtype[row] == "Ri":
                # Internal resistor
                desc.append(rtype[row] + str(num_Ri))
                num_Ri += 1
                val = Ri
            else:
                # Voltage source
                desc.append("V" + str(num_V))
                num_V += 1
                val = V
            node1.append(nodes[row + 1])
            node2.append(nodes[row])
            value.append(val)
            # netlist.append(netline)

    # +ve busbar (top row of the grid)
    if configuration == "parallel-strings":
        bus_nodes = [grid[-1, :]]
    elif configuration == "series-groups":
        bus_nodes = grid[3::3, :]
    else:
        raise ValueError("configuration must be parallel-strings or series-groups")

    for nodes in bus_nodes:
        for i in range(len(nodes) - 1):
            # netline = []
            desc.append("Rbp" + str(num_Rb))
            num_Rb += 1
            node1.append(nodes[i])
            node2.append(nodes[i + 1])
            value.append(Rb)

    desc = np.asarray(desc)
    node1 = np.asarray(node1)
    node2 = np.asarray(node2)
    value = np.asarray(value)
    main_grid = {
        "desc": desc,
        "node1": node1,
        "node2": node2,
        "value": value,
        "node1_x": x[node1 - 1],
        "node1_y": y[node1 - 1],
        "node2_x": x[node2 - 1],
        "node2_y": y[node2 - 1],
    }

    # Current source - spans the entire pack
    if (terminals == "left") or (terminals is None):
        t_nodes = [0, 0]
    elif terminals == "right":
        t_nodes = [-1, -1]
    elif terminals == "left-right":
        t_nodes = [0, -1]
    elif terminals == "right-left":
        t_nodes = [-1, 0]
    elif isinstance(terminals, (list, np.ndarray)):
        t_nodes = terminals
    else:
        raise ValueError(
            'Please specify a valid terminals argument: "left", '
            + '"right", "left-right" or "right-left" or a list or '
            + "array of nodes"
        )
    # terminal nodes
    t1 = grid[-1, t_nodes[0]]
    t2 = grid[0, t_nodes[1]]
    # terminal coords
    x1 = x[t1 - 1]
    x2 = x[t2 - 1]
    y1 = y[t1 - 1]
    y2 = y[t2 - 1]
    nn = grid.max() + 1  # next node
    # coords of nodes forming current source loop
    if terminals == "left" or (
        isinstance(terminals, (list, np.ndarray)) and np.all(np.array(terminals) == 0)
    ):
        ix = x1 - 1
        dy = 0
    elif terminals == "right" or (
        isinstance(terminals, (list, np.ndarray)) and np.all(np.array(terminals) == -1)
    ):
        ix = x1 + 1
        dy = 0
    else:
        ix = -1
        dy = 1
    if dy == 0:
        desc = ["Rtp1", "I0", "Rtn1"]
        xs = np.array([x1, ix, ix, x2])
        ys = np.array([y1, y1, y2, y2])
        node1 = [t1, nn, 0]
        node2 = [nn, 0, t2]
        value = [Rt, I, Rt]
        num_elem = 3
    else:
        desc = ["Rtp0", "Rtp1", "I0", "Rtn1", "Rtn0"]
        xs = np.array([x1, x1, ix, ix, x2, x2])
        ys = np.array([y1, y1 + dy, y1 + dy, 0 - dy, 0 - dy, y2])
        node1 = [t1, nn, nn + 1, 0, nn + 2]
        node2 = [nn, nn + 1, 0, nn + 2, t2]
        hRt = Rt / 2
        value = [hRt, hRt, I, hRt, hRt]
        num_elem = 5

    desc = np.asarray(desc)
    node1 = np.asarray(node1)
    node2 = np.asarray(node2)
    value = np.asarray(value)
    current_loop = {
        "desc": desc,
        "node1": node1,
        "node2": node2,
        "value": value,
        "node1_x": xs[:num_elem],
        "node1_y": ys[:num_elem],
        "node2_x": xs[1:],
        "node2_y": ys[1:],
    }

    for key in main_grid.keys():
        main_grid[key] = np.concatenate((main_grid[key], current_loop[key]))
    netlist = pd.DataFrame(main_grid)

    if plot:
        lp.simple_netlist_plot(netlist)
    lp.logger.notice("Circuit created")
    return netlist


def solve_circuit(netlist):
    """
    Generate and solve the Modified Nodal Analysis (MNA) equations for the circuit.
    The MNA equations are a linear system Ax = z.
    See http://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA3.html

    Args:
        netlist (pandas.DataFrame):
            A netlist of circuit elements with format desc, node1, node2, value.

    Returns:
        (np.ndarray, np.ndarray):
        - V_node: Voltages of the voltage elements
        - I_batt: Currents of the current elements

    """
    timer = pybamm.Timer()

    desc = np.array(netlist["desc"]).astype("<U16")
    node1 = np.array(netlist["node1"])
    node2 = np.array(netlist["node2"])
    value = np.array(netlist["value"])
    nLines = netlist.shape[0]

    n = np.concatenate((node1, node2)).max()  # Number of nodes (highest node number)

    m = 0  # "m" is the number of voltage sources, determined below.
    V_elem = ["V", "O", "E", "H"]
    for nm in desc:
        if nm[0] in V_elem:
            m += 1

    # Construct the A matrix, which will be a (n+m) x (n+m) matrix
    # A = [G    B]
    #     [B.T  D]
    # G matrix tracks the conductance between nodes (consists of floats)
    # B matrix tracks voltage sources between nodes (consists of -1, 0, 1)
    # D matrix is always zero for non-dependent sources
    # Construct the z vector with length (n+m)
    # z = [i]
    #     [e]
    # i is currents and e is voltages
    # Use lil matrices to construct the A array
    G = sp.sparse.lil_matrix((n, n))
    B = sp.sparse.lil_matrix((n, m))
    D = sp.sparse.lil_matrix((m, m))
    i = np.zeros([n, 1])
    e = np.zeros([m, 1])

    """
    % We need to keep track of the number of voltage sources we've parsed
    % so far as we go through file.  We start with zero.
    """
    vsCnt = 0
    """
    % This loop does the bulk of filling in the arrays.  It scans line by line
    % and fills in the arrays depending on the type of element found on the
    % current line.
    % See http://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA3.html
    """

    for k1 in range(nLines):
        n1 = node1[k1] - 1  # get the two node numbers in python index format
        n2 = node2[k1] - 1
        elem = desc[k1][0]
        if elem == "R":
            # Resistance elements: fill the G matrix only
            g = 1 / value[k1]  # conductance = 1 / R
            """
            % Here we fill in G array by adding conductance.
            % The procedure is slightly different if one of the nodes is
            % ground, so check for those accordingly.
            """
            if n1 == -1:  # -1 is the ground node
                G[n2, n2] = G[n2, n2] + g
            elif n2 == -1:
                G[n1, n1] = G[n1, n1] + g
            else:
                G[n1, n1] = G[n1, n1] + g
                G[n2, n2] = G[n2, n2] + g
                G[n1, n2] = G[n1, n2] - g
                G[n2, n1] = G[n2, n1] - g
        elif elem == "V":
            # Voltage elements: fill the B matrix and the e vector
            if n1 >= 0:
                B[n1, vsCnt] = B[n1, vsCnt] + 1
            if n2 >= 0:
                B[n2, vsCnt] = B[n2, vsCnt] - 1
            e[vsCnt] = value[k1]
            vsCnt += 1

        elif elem == "I":
            # Current elements: fill the i vector only
            if n1 >= 0:
                i[n1] = i[n1] - value[k1]
            if n2 >= 0:
                i[n2] = i[n2] + value[k1]

    # Construct final matrices from sub-matrices
    upper = sp.sparse.hstack((G, B))
    lower = sp.sparse.hstack((B.T, D))
    A = sp.sparse.vstack((upper, lower))
    # Convert a to csr sparse format for more efficient solving of the linear system
    # csr works slighhtly more robustly than csc
    A_csr = sp.sparse.csr_matrix(A)
    z = np.vstack((i, e))

    toc_setup = timer.time()
    lp.logger.debug(f"Circuit set up in {toc_setup}")

    # Scipy
    # X = solve(A, z).flatten()
    X = sp.sparse.linalg.spsolve(A_csr, z).flatten()
    # Pypardiso
    # X = pypardiso.spsolve(Aspr, z).flatten()

    # amg
    # ml = pyamg.smoothed_aggregation_solver(Aspr)
    # X = ml.solve(b=z, tol=1e-6, maxiter=10, accel="bicgstab")

    # include ground node (0V)
    # it is counter-intuitive that z is [i,e] while X is [V,I], but this is correct
    V_node = np.zeros(n + 1)
    V_node[1:] = X[:n]
    I_batt = X[n:]

    toc = timer.time()
    lp.logger.debug(f"Circuit solved in {toc - toc_setup}")
    lp.logger.info(f"Circuit set up and solved in {toc}")

    return V_node, I_batt


def solve_circuit_vectorized(netlist):
    """
    Generate and solve the Modified Nodal Analysis (MNA) equations for the circuit.
    The MNA equations are a linear system Ax = z.
    See http://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA3.html

    Args:
        netlist (pandas.DataFrame):
            A netlist of circuit elements with format desc, node1, node2, value.

    Returns:
        (np.ndarray, np.ndarray):
        - V_node: Voltages of the voltage elements
        - I_batt: Currents of the current elements
    """
    timer = pybamm.Timer()

    desc = np.array(netlist["desc"]).astype("<U1")  # just take first character
    node1 = np.array(netlist["node1"])
    node2 = np.array(netlist["node2"])
    value = np.array(netlist["value"])
    n = np.concatenate((node1, node2)).max()  # Number of nodes (highest node number)

    m = np.sum(desc == "V")  # we only use V in liionpack

    # Construct the A matrix, which will be a (n+m) x (n+m) matrix
    # A = [G    B]
    #     [B.T  D]
    # G matrix tracks the conductance between nodes (consists of floats)
    # B matrix tracks voltage sources between nodes (consists of -1, 0, 1)
    # D matrix is always zero for non-dependent sources
    # Construct the z vector with length (n+m)
    # z = [i]
    #     [e]
    # i is currents and e is voltages
    # Use lil matrices to construct the A array
    G = sp.sparse.lil_matrix((n, n))
    B = sp.sparse.lil_matrix((n, m))
    D = sp.sparse.lil_matrix((m, m))
    i = np.zeros([n, 1])
    e = np.zeros([m, 1])

    """
    % This old loop is now vectorized
    """

    node1 = node1 - 1  # get the two node numbers in python index format
    node2 = node2 - 1
    # Resistance elements: fill the G matrix only
    g = np.ones(len(value)) * np.nan
    n1_ground = node1 == -1
    n2_ground = node2 == -1
    # Resistors
    R_map = desc == "R"
    g[R_map] = 1 / value[R_map]  # conductance = 1 / R
    R_map_n1_ground = np.logical_and(R_map, n1_ground)
    R_map_n2_ground = np.logical_and(R_map, n2_ground)
    R_map_ok = np.logical_and(R_map, ~np.logical_or(n1_ground, n2_ground))

    """
    % Here we fill in G array by adding conductance.
    % The procedure is slightly different if one of the nodes is
    % ground, so check for those accordingly.
    """
    if np.any(R_map_n1_ground):  # -1 is the ground node
        n2 = node2[R_map_n1_ground]
        G[n2, n2] = G[n2, n2] + g[R_map_n1_ground]
    if np.any(R_map_n2_ground):
        n1 = node1[R_map_n2_ground]
        G[n1, n1] = G[n1, n1] + g[R_map_n2_ground]

    # No longer needs unique nodes
    # We can take advantage of the fact that coo style inputs sum
    # duplicates with converted to csr
    n1 = node1[R_map_ok]
    n2 = node2[R_map_ok]
    g_change = g[R_map_ok]
    gn1n1 = sp.sparse.csr_matrix((g_change, (n1, n1)), shape=G.shape)
    gn2n2 = sp.sparse.csr_matrix((g_change, (n2, n2)), shape=G.shape)
    gn1n2 = sp.sparse.csr_matrix((g_change, (n1, n2)), shape=G.shape)
    gn2n1 = sp.sparse.csr_matrix((g_change, (n2, n1)), shape=G.shape)
    G += gn1n1
    G += gn2n2
    G -= gn1n2
    G -= gn2n1

    # Assume Voltage sources do not connect directly to ground
    V_map = desc == "V"
    V_map_not_n1_ground = np.logical_and(V_map, ~n1_ground)
    V_map_not_n2_ground = np.logical_and(V_map, ~n2_ground)
    n1 = node1[V_map_not_n1_ground]
    n2 = node2[V_map_not_n2_ground]
    vsCnt = np.ones(len(V_map)) * -1
    vsCnt[V_map] = np.arange(m)
    # Voltage elements: fill the B matrix and the e vector
    B[n1, vsCnt[V_map_not_n1_ground]] = 1
    B[n2, vsCnt[V_map_not_n2_ground]] = -1
    e[np.arange(m), 0] = value[V_map]
    # Current Sources
    I_map = desc == "I"
    n1 = node1[I_map]
    n2 = node2[I_map]
    # Current elements: fill the i vector only
    if n1 >= 0:
        i[n1] = i[n1] - value[I_map]
    if n2 >= 0:
        i[n2] = i[n2] + value[I_map]

    # Construct final matrices from sub-matrices
    upper = sp.sparse.hstack((G, B))
    lower = sp.sparse.hstack((B.T, D))
    A = sp.sparse.vstack((upper, lower))
    # Convert a to csr sparse format for more efficient solving of the linear system
    # csr works slighhtly more robustly than csc
    A_csr = sp.sparse.csr_matrix(A)
    z = np.vstack((i, e))

    toc_setup = timer.time()
    lp.logger.debug(f"Circuit set up in {toc_setup}")

    # Scipy
    X = sp.sparse.linalg.spsolve(A_csr, z).flatten()

    # include ground node (0V)
    # it is counter-intuitive that z is [i,e] while X is [V,I], but this is correct
    V_node = np.zeros(n + 1)
    V_node[1:] = X[:n]
    I_batt = X[n:]

    toc = timer.time()
    lp.logger.debug(f"Circuit solved in {toc - toc_setup}")
    lp.logger.info(f"Circuit set up and solved in {toc}")

    return V_node, I_batt


def make_lcapy_circuit(netlist):
    """
    Generate a circuit that can be used with lcapy

    Args:
        netlist (pandas.DataFrame):
            A netlist of circuit elements with format. desc, node1, node2, value.

    Returns:
        lcapy.Circuit:
            The Circuit class is used for describing networks using netlists.
            Despite the name, it does not require a closed path.

    """
    cct = Circuit()
    I_map = netlist["desc"].str.find("I") > -1
    net2 = netlist.copy()
    net2.loc[I_map, ("node1")] = netlist["node2"][I_map]
    net2.loc[I_map, ("node2")] = netlist["node1"][I_map]
    d1 = "down"
    d2 = "up"
    I_xs = [net2[I_map]["node1_x"].values[0], net2[I_map]["node2_x"].values[0]]
    I_left = np.any(np.array(I_xs) == -1)
    all_desc = netlist["desc"].values
    for index, row in net2.iterrows():
        color = "black"
        desc, n1, n2, value, n1x, n1y, n2x, n2y = row[:8]
        if desc[0] == "V":
            direction = d1
        elif desc[0] == "I":
            direction = d2
        elif desc[0] == "R":
            if desc[1] == "b":
                direction = "right"
            elif desc[1] == "t":
                # These are the terminal nodes and require special attention
                if desc[2] == "p":
                    # positive
                    color = "red"
                else:
                    # negative
                    color = "blue"
                # If terminals are not both at the same end then the netlist
                # has two resistors with half the value to make a nice circuit
                # diagram. Convert into 1 resistor + 1 wire
                if desc[3] == "0":
                    # The wires have the zero suffix
                    direction = d2
                    desc = "W"
                else:
                    # The reistors have the 1 suffix
                    # Convert the value to the total reistance if a wire element
                    # is in the netlist
                    w_desc = desc[:3] + "0"
                    if w_desc in all_desc:
                        value *= 2
                    desc = desc[:3]
                    # Terminal loop is C shaped with positive at the top so
                    # order is left-vertical-right if we're on the left side
                    # and right-vertical-left if we're on the right side
                    if desc[2] == "p":
                        if I_left:
                            direction = "left"
                            # if the terminal connection is not at the end then
                            # extend the element connections
                            if n1x > 0:
                                direction += "=" + str(1 + n1x)
                        else:
                            direction = "right"
                            if n1x < I_xs[0] - 1:
                                direction += "=" + str(1 + I_xs[0] - n1x)
                    else:
                        if I_left:
                            direction = "right"
                        else:
                            direction = "left"
            else:
                direction = d1
        if desc == "W":
            string = desc + " " + str(n1) + " " + str(n2)
        else:
            string = desc + " " + str(n1) + " " + str(n2) + " " + str(value)
        string = string + "; " + direction
        string = string + ", color=" + color
        cct.add(string)
    # Add ground node
    cct.add("W 0 00; down, sground")
    return cct


def power_loss(netlist, include_Ri=False):
    """
    Calculate the power loss through joule heating of all the resistors in the
    circuit

    Args:
        netlist (pandas.DataFrame):
            A netlist of circuit elements with format desc, node1, node2, value.
        include_Ri (bool):
            Default is False. If True the internal resistance of the batteries
            is included

    Returns:
        None

    """
    V_node, I_batt = lp.solve_circuit_vectorized(netlist)
    R_map = netlist["desc"].str.find("R") > -1
    R_map = R_map.values
    if not include_Ri:
        Ri_map = netlist["desc"].str.find("Ri") > -1
        Ri_map = Ri_map.values
        R_map *= ~Ri_map
    R_value = netlist[R_map].value.values
    R_node1 = netlist[R_map].node1.values
    R_node2 = netlist[R_map].node2.values
    R_node1_V = V_node[R_node1]
    R_node2_V = V_node[R_node2]
    V_diff = np.abs(R_node1_V - R_node2_V)
    P_loss = V_diff**2 / R_value
    netlist["power_loss"] = 0
    netlist.loc[R_map, ("power_loss")] = P_loss


def _fn(n):
    if n == 0:
        return "0"
    else:
        return "N" + str(n).zfill(3)


def write_netlist(netlist, filename):
    """
    Write netlist to file

    Args:
        netlist (pandas.DataFrame):
            A netlist of circuit elements with format desc, node1, node2, value.

    Returns:
        None


    """
    lines = ["* " + filename]
    for i, r in netlist.iterrows():
        line = r.desc + " " + _fn(r.node1) + " " + _fn(r.node2) + " " + str(r.value)
        lines.append(line)
    lines.append(".op")
    lines.append(".backanno")
    lines.append(".end")
    with open(filename, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
