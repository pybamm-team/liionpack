# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:36:15 2021

@author: Tom
"""
import numpy as np
import codecs
import pandas as pd
import time as ticker
import matplotlib.pyplot as plt
import liionpack as lp
import os
# from scipy.linalg import solve
# import pyamg
# import pypardiso
import pybamm
import scipy as sp

def read_netlist(filepath, Ri=1e-2, Rc=1e-2, Rb=1e-4, Rl=5e-4, I=80.0, V=4.2):
    r'''
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
        A netlist of circuit elements with format. desc, node1, node2, value.
        

    '''

    if '.cir' not in filepath:
        filepath += '.cir'
    if not os.path.isfile(filepath):
        temp = os.path.join(lp.CIRCUIT_DIR, filepath)
        if not os.path.isfile(temp):
            pass
        else:
            filepath = temp
    with codecs.open(filepath, "r", "utf-16LE") as fd:
        Lines = fd.readlines()
    Lines = [l.strip('\n').split(' ') for l in Lines if l[0] not in ['*', '.']]
    Lines = np.array(Lines, dtype='<U16')
    desc = Lines[:, 0]
    N1 = Lines[:, 1]
    N2 = Lines[:, 2]
    # values = Lines[:, 3]
    values = np.zeros(len(N1))
    N1 = np.array([x.strip('N') for x in N1], dtype=int)
    N2 = np.array([x.strip('N') for x in N2], dtype=int)
    netlist = pd.DataFrame({'desc': desc, 'node1': N1, 'node2': N2, 'value': values})
    Ri_map = netlist['desc'].str.find('Ri') > -1
    Rc_map = netlist['desc'].str.find('Rc') > -1
    Rb_map = netlist['desc'].str.find('Rb') > -1
    Rl_map = netlist['desc'].str.find('Rl') > -1
    V_map = netlist['desc'].str.find('V') > -1
    I_map = netlist['desc'].str.find('I') > -1
    netlist.loc[Ri_map, ('value')] = Ri
    netlist.loc[Rc_map, ('value')] = Rc
    netlist.loc[Rb_map, ('value')] = Rb
    netlist.loc[Rl_map, ('value')] = Rl
    netlist.loc[I_map, ('value')] = I
    netlist.loc[V_map, ('value')] = V
    pybamm.logger.notice('netlist ' + filepath + ' loaded')
    return netlist


def _make_contiguous(node1, node2):
    r'''
    
    Internal helper function to make the netlist nodes contiguous

    Parameters
    ----------
    node1 : int
        First node in the netlist.
    node2 : int
        Second node in the netlist.

    Returns
    -------
    int
        First nodes.
    int
        Second nodes.

    '''
    nodes = np.vstack((node1, node2)).astype(int)
    nodes = nodes.T
    unique_nodes = np.unique(nodes)
    nodes_copy = nodes.copy()
    for i in range(len(unique_nodes)):
        nodes_copy[nodes == unique_nodes[i]] = i

    return nodes_copy[:, 0], nodes_copy[:, 1]


def setup_circuit(Np=1, Ns=1, Ri=1e-2, Rc=1e-2, Rb=1e-4, Rl=5e-4, I=80.0, V=4.2, plot=False):
    r'''
    
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
    netlist : TYPE
        DESCRIPTION.

    '''
    Nc = Np + 1
    Nr = Ns * 3 + 1

    grid = np.arange(Nc * Nr).reshape([Nr, Nc])

    # 1st column is terminals only
    # 1st and last rows are busbars
    # Other rows alternate between series resistor and voltage source

    # Build data  with ['element type', Node1, Node2, value]
    netlist = []

    num_Rb = 0
    num_I = 0
    num_V = 0

    desc = []
    node1 = []
    node2 = []
    value = []
    

    # -ve busbars
    bus_nodes = [grid[-1, :]]
    for nodes in bus_nodes:
        for i in range(len(nodes) - 1):
            # netline = []
            desc.append('Rbn' + str(num_Rb))
            num_Rb += 1
            node1.append(nodes[i])
            node2.append(nodes[i + 1])
            value.append(Rb)
    num_Rs = 0
    num_Ri = 0
    # Series resistors and voltage sources
    cols = np.arange(Nc)[1:]
    rows = np.arange(Nr)[:-1]
    rtype = ['Rs', 'V', 'Ri']*Ns
    for col in cols:
        nodes = grid[:, col]
        for row in rows:
            # netline = []
            if rtype[row][0] == 'R':
                if rtype[row][1] == 's':
                    # Series resistor
                    desc.append(rtype[row] + str(num_Rs))
                    num_Rs += 1
                    val = Rc
                else:
                    # Internal resistor
                    desc.append(rtype[row] + str(num_Ri))
                    num_Ri += 1
                    val = Ri
                node1.append(nodes[row])
                node2.append(nodes[row + 1])
            else:
                # Voltage source
                desc.append('V' + str(num_V))
                num_V += 1
                val = V
                node1.append(nodes[row + 1])
                node2.append(nodes[row])
            value.append(val)
            # netlist.append(netline)

    # +ve busbar
    bus_nodes = [grid[0, :]]
    for nodes in bus_nodes:
        for i in range(len(nodes) - 1):
            # netline = []
            desc.append('Rbp' + str(num_Rb))
            num_Rb += 1
            node1.append(nodes[i + 1])
            node2.append(nodes[i])
            value.append(Rb)

    # Current source - same end
    # netline = []
    # terminal_nodes = [, ]
    desc.append('I' + str(num_I))
    num_I += 1
    node1.append(grid[0, 0])
    node2.append(grid[-1, 0])
    value.append(I)


    coords = np.indices(grid.shape)
    y = coords[0, :, :].flatten()
    x = coords[1, :, :].flatten()
    if plot:
        plt.figure()
        # plt.scatter(x, y, c='k')
        for netline in zip(desc, node1, node2):
            elem, n1, n2, = netline
            if elem[0] == 'I':
                color = 'g'
            elif elem[0] == 'R':
                if elem[1] == 's':
                    color = 'r'
                elif elem[1] == 'b':
                    color = 'k'
                else:
                    color = 'y'
            elif elem[0] == 'V':
                color = 'b'
            x1 = x[n1]
            x2 = x[n2]
            y1 = y[n1]
            y2 = y[n2]
            plt.scatter([x1, x2], [y1, y2], c='k')
            plt.plot([x1, x2], [y1, y2], c=color)

    desc = np.asarray(desc)
    node1 = np.asarray(node1)
    node2 = np.asarray(node2)
    value = np.asarray(value)

    node1, node2 = _make_contiguous(node1, node2)
    netlist = pd.DataFrame({'desc': desc, 'node1': node1,
                            'node2': node2, 'value': value})

    pybamm.logger.notice("Circuit created")
    return netlist

def solve_circuit(netlist):
    r'''
    

    Parameters
    ----------
    netlist : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    tic = ticker.time()

    Name = np.array(netlist['desc']).astype('<U16')
    N1 = np.array(netlist['node1'])
    N2 = np.array(netlist['node2'])
    arg3 = np.array(netlist['value'])
    n = np.concatenate((N1, N2)).max()  # Highest node number
    nLines = netlist.shape[0]
    m = 0  # "m" is the number of voltage sources, determined below.
    V_elem = ['V', 'O', 'E', 'H']
    for nm in Name:
        if nm[0] in V_elem:
            m += 1

    G = np.zeros([n, n])
    B = np.zeros([n, m])
    D = np.zeros([m, m])
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
        n1 = N1[k1] - 1  # get the two node numbers in python index format
        n2 = N2[k1] - 1
        elem = Name[k1][0]
        if elem == 'R':
            g = 1 / arg3[k1]  # conductance = 1 / R
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
        elif elem == 'V':
            if n1 >= 0:
                B[n1, vsCnt] = B[n1, vsCnt] + 1
            if n2 >= 0:
                B[n2, vsCnt] = B[n2, vsCnt] - 1
            e[vsCnt] = arg3[k1]
            vsCnt += 1

        elif elem == 'I':
            if n1 >= 0:
                i[n1] = i[n1] - arg3[k1]
            if n2 >= 0:
                i[n2] = i[n2] + arg3[k1]

    upper = np.hstack((G, B))
    lower = np.hstack((B.T, D))
    A = np.vstack((upper, lower))
    z = np.vstack((i, e))
    Aspr = sp.sparse.csr_matrix(A)
    # Scipy
    # X = solve(A, z).flatten()
    X = sp.sparse.linalg.spsolve(Aspr, z).flatten()
    
    # Pypardiso
    # X = pypardiso.spsolve(Aspr, z).flatten()
    
    # amg
    # ml = pyamg.smoothed_aggregation_solver(Aspr)
    # X = ml.solve(b=z, tol=1e-6, maxiter=10, accel="bicgstab")

    # include ground node
    V_node = np.zeros(n + 1)
    V_node[1:] = X[:n]
    I_batt = X[n:]

    toc = ticker.time()
    pybamm.logger.info("Circuit solved in " +
                       str(np.around(toc-tic, 3)) + " s")
    return V_node, I_batt