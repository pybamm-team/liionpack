# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:36:15 2021

@author: Tom
"""
import numpy as np
import codecs
import pandas as pd
import matplotlib.pyplot as plt
import liionpack as lp
import os

# from scipy.linalg import solve
# import pyamg
# import pypardiso
import pybamm
import scipy as sp


def read_netlist(filepath, Ri=1e-2, Rc=1e-2, Rb=1e-4, Rl=5e-4, I=80.0, V=4.2):
    r"""
    Assumes netlist has been saved by LTSpice with format Descriptor Node1 Node2 Value
    Any lines starting with * are comments and . are commands so ignore them
    Nodes begin with N so remove that
    Open ended components are not allowed and their nodes start with NC (no-connection)

    Parameters
    ----------
    filepath : str
        path to netlist circuit file '.cir'.
    Ri : float
        Internal resistance (:math:`\Omega`). The default is 1e-2.
    Rc : float
        Connection resistance (:math:`\Omega`). The default is 1e-2.
    Rb : float
        Busbar resistance (:math:`\Omega`). The default is 1e-4.
    Rl : float
        Long Busbar resistance (:math:`\Omega`). The default is 5e-4.
    I : float
        Current (A). The default is 80.0.
    V : float
        Initial battery voltage (V). The default is 4.2.

    Returns
    -------
    netlist : pandas.DataFrame
        A netlist of circuit elements with format desc, node1, node2, value.


    """

    # Read in the netlist
    if ".cir" not in filepath:
        filepath += ".cir"
    if not os.path.isfile(filepath):
        temp = os.path.join(lp.CIRCUIT_DIR, filepath)
        if not os.path.isfile(temp):
            pass
        else:
            filepath = temp
    with codecs.open(filepath, "r", "utf-16LE") as fd:
        Lines = fd.readlines()
    # Ignore lines starting with * or .
    Lines = [l.strip("\n").split(" ") for l in Lines if l[0] not in ["*", "."]]
    Lines = np.array(Lines, dtype="<U16")

    # Read descriptions and nodes, strip N from nodes
    # Lines is desc | node1 | node2
    desc = Lines[:, 0]
    node1 = Lines[:, 1]
    node2 = Lines[:, 2]
    value = np.zeros(len(node1))
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
        ("Rl", Rl),
        ("I", I),
        ("V", V),
    ]:
        # netlist["desc"] consists of entries like 'Ri13'
        # this map finds all the entries that start with (e.g.) 'Ri'
        name_map = netlist["desc"].str.find(name) > -1
        # then allocates the value to the corresponding indices
        netlist.loc[name_map, ("value")] = val

    lp.logger.notice("netlist " + filepath + " loaded")
    return netlist


def _make_contiguous(node1, node2):
    r"""

    Internal helper function to make the netlist nodes contiguous

    Parameters
    ----------
    node1 : array
        First node in the netlist.
    node2 : array
        Second node in the netlist.

    Returns
    -------
    array
        First nodes.
    array
        Second nodes.

    """
    nodes = np.vstack((node1, node2)).astype(int)
    nodes = nodes.T
    unique_nodes = np.unique(nodes)
    nodes_copy = nodes.copy()
    for i in range(len(unique_nodes)):
        nodes_copy[nodes == unique_nodes[i]] = i

    return nodes_copy[:, 0], nodes_copy[:, 1]


def setup_circuit(
    Np=1, Ns=1, Ri=1e-2, Rc=1e-2, Rb=1e-4, Rl=5e-4, I=80.0, V=4.2, plot=False
):
    r"""

    Define a netlist from a number of batteries in parallel and series

    Parameters
    ----------
    Np : int
        Number of batteries in parallel. The default is 1.
    Ns : int
        Number of batteries in series. The default is 1.
    Ri : float
        Internal resistance (:math:`\Omega`). The default is 1e-2.
    Rc : float
        Connection resistance (:math:`\Omega`). The default is 1e-2.
    Rb : float
        Busbar resistance (:math:`\Omega`). The default is 1e-4.
    I : float
        Current (A). The default is 80.0.
    V : float
        Initial battery voltage (V). The default is 4.2.
    plot : bool, optional
        Plot the circuit. The default is False.

    Returns
    -------
    netlist : pandas.DataFrame
        A netlist of circuit elements with format desc, node1, node2, value.

    """
    Nc = Np + 1
    Nr = Ns * 3 + 1

    grid = np.arange(Nc * Nr).reshape([Nr, Nc])
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
    num_I = 0
    num_V = 0

    desc = []
    node1 = []
    node2 = []
    value = []

    # -ve busbars (final row of the grid)
    bus_nodes = [grid[-1, :]]
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
    cols = np.arange(Nc)[1:]
    rows = np.arange(Nr)[:-1]
    rtype = ["Rs", "V", "Ri"] * Ns
    for col in cols:
        # Go down the column alternating Rs, V, Ri connections between nodes
        nodes = grid[:, col]
        for row in rows:
            if rtype[row] == "Rs":
                # Series resistor
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
            if rtype[row][0] == "R":
                node1.append(nodes[row + 1])
                node2.append(nodes[row])
            else:
                node1.append(nodes[row + 1])
                node2.append(nodes[row])               
            value.append(val)
            # netlist.append(netline)

    # +ve busbar (first row of the grid)
    bus_nodes = [grid[0, :]]
    for nodes in bus_nodes:
        for i in range(len(nodes) - 1):
            # netline = []
            desc.append("Rbp" + str(num_Rb))
            num_Rb += 1
            node1.append(nodes[i + 1])
            node2.append(nodes[i])
            value.append(Rb)

    # Current source - spans the entire first column
    desc.append("I" + str(num_I))
    num_I += 1
    node1.append(grid[-1, 0])
    node2.append(grid[0, 0])
    value.append(I)

    coords = np.indices(grid.shape)
    y = coords[0, :, :].flatten()
    x = coords[1, :, :].flatten()
    if plot:
        plt.figure()
        for netline in zip(desc, node1, node2):
            (
                elem,
                n1,
                n2,
            ) = netline
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
            x1 = x[n1]
            x2 = x[n2]
            y1 = y[n1]
            y2 = y[n2]
            plt.scatter([x1, x2], [y1, y2], c="k")
            plt.plot([x1, x2], [y1, y2], c=color)

    desc = np.asarray(desc)
    node1 = np.asarray(node1)
    node2 = np.asarray(node2)
    value = np.asarray(value)

    node1, node2 = _make_contiguous(node1, node2)
    netlist = pd.DataFrame(
        {"desc": desc, "node1": node1, "node2": node2, "value": value}
    )

    lp.logger.notice("Circuit created")
    return netlist


def solve_circuit(netlist):
    r"""
    Generate and solve the Modified Nodal Analysis (MNA) equations for the circuit.
    The MNA equations are a linear system Ax = z.
    See http://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA3.html

    Parameters
    ----------
    netlist : pandas.DataFrame
        A netlist of circuit elements with format desc, node1, node2, value.

    Returns
    -------
    V_node : array
        Voltages of the voltage elements
    I_batt : array
        Currents of the current elements

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
