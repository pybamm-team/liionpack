from lcapy import Circuit

def draw_circuit(netlist):
    r'''
    Draw a latex version of netlist circuit

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
    
    cct.draw(node_spacing=3.0, dpi=300)
