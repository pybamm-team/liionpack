# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:44:19 2021

@author: Tom
"""

from numpy import exp


def generate_protocol_from_experiment(experiment):
    r'''
    

    Parameters
    ----------
    experiment : pybamm.Experiment class
        The experiment to generate the protocol from.

    Returns
    -------
    proto : list
        a sequence of terminal currents to apply at each timestep

    '''
    proto = []
    for op in experiment.operating_conditions:
        t = op["time"]
        dt = op["period"]
        if t % dt != 0:
            raise ValueError("Time must be a multiple of the period")
        I, typ = op["electric"]
        if typ != "A":
            raise ValueError("Only constant current operations are supported")
        proto.extend([I] * int(t / dt))
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