from lcapy import Circuit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sympy import init_printing

init_printing(pretty_print=False)

def draw_circuit(netlist, **kwargs):
    r'''
    Draw a latex version of netlist circuit
    N.B only works with generated netlists not imported ones.

    Parameters
    ----------
    netlist : pandas.DataFrame
        A netlist of circuit elements with format. desc, node1, node2, value.

    Returns
    -------
    None.


    Example
    >>> import liionpack as lp
    >>> net = lp.setup_circuit(Np=3, Ns=1, Rb=1e-4, Rc=1e-2, Ri=5e-2, V=3.2, I=80.0)
    >>> lp.draw_circuit(net)
    '''
    cct = Circuit()
    V_map = netlist['desc'].str.find('V') > -1
    I_map = netlist['desc'].str.find('I') > -1
    net2 = netlist.copy()
    net2.loc[V_map, ('node1')] = netlist['node2'][V_map]
    net2.loc[V_map, ('node2')] = netlist['node1'][V_map]
    net2.loc[I_map, ('node1')] = netlist['node2'][I_map]
    net2.loc[I_map, ('node2')] = netlist['node1'][I_map]
    
    for index, row in net2.iterrows():
        # print(row['desc'])
        string = ""
        direction = ""
        for col in row.iteritems():
            if col[0] == 'desc':
                if col[1][0] == 'V':
                    direction = 'down'
                elif col[1][0] == 'I':
                    direction = 'up'
                elif col[1][0] == 'R':
                    if col[1][1] == 'b':
                        if col[1][2] == 'n':
                            direction = 'right'
                        else:
                            direction = 'left'
                    else:
                        direction = 'down'
            string = string + str(col[1]) + " "
    
        string = string + '; ' + direction
        cct.add(string)

    cct.draw(**kwargs)

def _text_color(vals,vmin,vmax,cmap):
    r'''
    

    Parameters
    ----------
    vals : TYPE
        DESCRIPTION.
    vmin : TYPE
        DESCRIPTION.
    vmax : TYPE
        DESCRIPTION.
    cmap : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    '''
    # return list of either black or white to write text, depending on whether
    # plotted color is closer to white or black
    cm = mpl.cm.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    val_cm = cm(norm(vals))[:,:3]
    val_norm = np.dot(val_cm, [0.2989, 0.5870, 0.1140])
    return ['k' if v>0.5 else 'w' for v in val_norm]

def _cell_text(ax, X, Y, vals, prec, text_colors):
    r'''
    

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    vals : TYPE
        DESCRIPTION.
    prec : TYPE
        DESCRIPTION.
    text_colors : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # X_pos, Y_pos = cell_XY_positions()
    for i, val in enumerate(vals):
        ax.text(x=X[i],y=Y[i],s="{:.{}f}".format(val,prec),color=text_colors[i], ha='center', va='center')

def _cell_text_numbers(ax, X, Y, text_colors):
    r'''
    

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    text_colors : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # X_pos, Y_pos = cell_XY_positions()
    y_offset = 0.005
    for i, [x_pos, y_pos] in enumerate(zip(X, Y)):
        ax.text(x=x_pos, y=y_pos - y_offset, s="{:d}".format(i+1), color=text_colors[i], ha='center', va='top',fontsize=7)

def cell_scatter_plot(ax, X, Y, c, text_prec=1, **kwargs):
    r"""

    Parameters
    ----------
    ax : matplotlib axis obj
        axis to plot on.
    X : float array
        x-coordinate of the battery
    Y : float array
        y-coordinate of the battery 
    c : like plt.scatter c kwarg
        colors to plot scatter with.
    text_prec : int
        precsition to write text of values on cells.
    **kwargs : 
        plt.scatter kwargs.

    Returns
    -------
    None.

    """
    
    # X_pos, Y_pos = cell_XY_positions()
    
    # set size of markers
    diameter = 21.44/1000;
    area = np.pi*(diameter/2)**2;
    s = area*4/np.pi # points 
    s = s*72/0.0254*1000 # some scaling... 
    
    # scatter plot
    sc = ax.scatter(X, Y, s, c, **kwargs)
    # set limits
    ax.set_xlim(-0.1,0.1)
    ax.set_ylim(-0.06,0.06)
    # colorbar
    plt.colorbar(sc, ax = ax, orientation = 'vertical')
    # set axis equal
    ax.set_aspect('equal')
    ax.set_axis_off()
    
    vmin, vmax = sc.get_clim()
    if 'cmap' in kwargs:
        cmap = kwargs.get('cmap')
    else:
        cmap='viridis'

    text_colors =  _text_color(c,vmin,vmax,cmap)
    # write cell text
    _cell_text(ax, X, Y, c, text_prec, text_colors)     
    _cell_text_numbers(ax, X, Y, text_colors)


def plot_pack(output):
    """
    Plot the battery pack voltage and current.

    Parameters
    ----------
    output : dict
        Output from liionpack.solve which contains pack and cell variables.
    """

    # Get pack level results
    time = output['Time [s]']
    v_pack = output['Pack terminal voltage [V]']
    i_pack = output['Pack current [A]']

    # Plot pack voltage and current
    _, ax = plt.subplots(tight_layout=True)
    ax.plot(time, v_pack, color='red', label='simulation')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Pack terminal voltage [V]', color='red')
    ax2 = ax.twinx()
    ax2.plot(time, i_pack, color='blue', label='simulation')
    ax2.set_ylabel('Pack current [A]', color='blue')
    ax2.set_title('Pack Summary')


def plot_cells(output):
    """
    Plot results for the battery cells.

    Parameters
    ----------
    output : dict
        Output from liionpack.solve which contains pack and cell variables.
    """

    # Get results for the battery cells
    time = output['Time [s]']
    v_cells = output['Terminal voltage [V]']
    i_cells = output['Cell current [A]']
    ocv_cells = output['Measured battery open circuit voltage [V]']
    ecm_cells = output['Local ECM resistance [Ohm]']
    heat_cells = output['X-averaged total heating [W.m-3]']
    temp_cells = output['Volume-averaged cell temperature [K]']
    negconc_cells = output['X-averaged negative particle surface concentration [mol.m-3]']
    posconc_cells = output['X-averaged positive particle surface concentration [mol.m-3]']

    # Get number of cells and create colormap
    n = len(v_cells[0])
    colors = plt.cm.jet(np.linspace(0, 1, n))

    # Plot voltages
    _, ax = plt.subplots(tight_layout=True)
    for i in range(n):
        ax.plot(time, v_cells[:, i], color=colors[i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Terminal voltage [V]')

    # Plot currents
    _, ax = plt.subplots(tight_layout=True)
    for i in range(n):
        ax.plot(time, i_cells[:, i], color=colors[i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Cell current [A]')

    # Plot open circuit voltages
    _, ax = plt.subplots(tight_layout=True)
    for i in range(n):
        ax.plot(time, ocv_cells[:, i], color=colors[i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Open circuit voltage [V]')

    # Plot resistances
    _, ax = plt.subplots(tight_layout=True)
    for i in range(n):
        ax.plot(time, ecm_cells[:, i], color=colors[i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('ECM resistance [Ω]')

    # Plot heating and temperatures
    _, (ax1, ax2) = plt.subplots(2, sharex=True, tight_layout=True)
    for i in range(n):
        ax1.plot(time, heat_cells[:, i], color=colors[i])
        ax2.plot(time, temp_cells[:, i], color=colors[i])
    ax1.set_ylabel('Total heating [W/m³]')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Temperature [K]')

    # Plot surface concentrations
    _, (ax1, ax2) = plt.subplots(2, sharex=True, tight_layout=True)
    for i in range(n):
        ax1.plot(time, negconc_cells[:, i], color=colors[i])
        ax2.plot(time, posconc_cells[:, i], color=colors[i])
    ax1.set_ylabel('Neg. surface conc. [mol/m³]')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Pos. surface conc. [mol/m³]')


def show_plots():
    """
    Wrapper function for the Matplotlib show() function.
    """
    plt.show()
