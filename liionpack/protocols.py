# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:44:19 2021

@author: Tom
"""

def generate_protocol(I_dch=-50, I_chg=50, t_dch=180, t_chg=180, t_rest=90,
                      chg_first=True):
    r'''
    

    Parameters
    ----------
    I_dch : float
        Negative discharge current [A]. The default is -50.
    I_chg : float
        Positive charge current [A]. The default is 50.
    t_dch : int
        number of time steps to discharge. The default is 180.
    t_chg : int
        number of time steps to charge. The default is 180.
    t_rest : int, optional
        number of time steps to rest inbetween charge and discharge.
        The default is 90.
    chg_first : bool
        charge before discharge. The default is True

    Returns
    -------
    proto : list
        a sequence of terminal currents to apply at each timestep

    '''
    if chg_first:
        proto = [I_chg] * t_chg + [0.0] * t_rest + [I_dch] * t_dch + [0.0] * t_rest
    else:
        proto = [I_dch] * t_dch + [0.0] * t_rest + [I_chg] * t_chg + [0.0] * t_rest
    return proto

def test_protocol():
    r'''
    retun a simple test protocol with no options

    Returns
    -------
    proto : list
        a test sequence of terminal currents to apply at each timestep.

    '''
    return generate_protocol(I_dch=-50, I_chg=50, t_dch=30, t_chg=30, t_rest=15)