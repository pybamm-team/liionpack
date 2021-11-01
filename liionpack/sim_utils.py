# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:41:16 2021

@author: Tom
"""
import pickle
import pybamm
import liionpack as lp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

init_fname = os.path.join(lp.INIT_FUNCS, "init_funcs.pickle")


def update_init_conc(sim, SoC=-1, OCV=-1):
    r"""


    Parameters
    ----------
    sim : TYPE
        DESCRIPTION.
    SoC : TYPE, optional
        DESCRIPTION. The default is -1.
    OCV : TYPE, optional
        DESCRIPTION. The default is -1.

    Returns
    -------
    None.

    """
    param = sim.parameter_values
    c_s_n_init, c_s_p_init = initial_conditions(SoC=SoC, OCV=OCV)
    param.update(
        {
            "Initial concentration in negative electrode [mol.m-3]": c_s_n_init,
            "Initial concentration in positive electrode [mol.m-3]": c_s_p_init,
        }
    )


def initial_conditions(SoC=-1, OCV=-1):
    r"""


    Parameters
    ----------
    SoC : TYPE, optional
        DESCRIPTION. The default is -1.
    OCV : TYPE, optional
        DESCRIPTION. The default is -1.

    Returns
    -------
    c_s_n_init : TYPE
        DESCRIPTION.
    c_s_p_init : TYPE
        DESCRIPTION.

    """

    with open(init_fname, "rb") as handle:
        init_funcs = pickle.load(handle)

    x_n_SoC = init_funcs["x_n_SoC"]
    x_p_SoC = init_funcs["x_p_SoC"]
    x_n_OCV = init_funcs["x_n_OCV"]
    x_p_OCV = init_funcs["x_p_OCV"]

    c_s_n_max = init_funcs["c_s_n_max"]
    c_s_p_max = init_funcs["c_s_p_max"]

    vmin = init_funcs["vmin"]
    vmax = init_funcs["vmax"]

    if SoC >= 0.0 and SoC <= 1.0:
        x_n_init = x_n_SoC(SoC)
        x_p_init = x_p_SoC(SoC)
    elif OCV >= vmin and OCV <= vmax:
        x_n_init = x_n_OCV(OCV)
        x_p_init = x_p_OCV(OCV)
    c_s_n_init = x_n_init * c_s_n_max
    c_s_p_init = x_p_init * c_s_p_max
    return c_s_n_init, c_s_p_init


def create_init_funcs(parameter_values=None):
    r"""


    Parameters
    ----------
    parameter_values : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    V_upper_limit = parameter_values["Upper voltage cut-off [V]"]
    V_lower_limit = parameter_values["Lower voltage cut-off [V]"]

    # Make sure we slow charge to upper limit and establish equilibrium
    experiment = pybamm.Experiment(
        [
            f"Charge at C/50 until {V_upper_limit}V (10 second period)",
            f"Hold at {V_upper_limit}V until 0.5mA (10 second period)",
            "Rest for 30 minutes (1 minute period)",
            f"Discharge at C/100 until {V_lower_limit}V (1 minute period)",
        ]
    )
    # sim.set_up_experiment(model, experiment)
    # sim = pybamm.Simulation(model=model, parameter_values=param, experiment=experiment)
    sim = lp.create_simulation(parameter_values=parameter_values, experiment=experiment)
    sim.solve()
    # sim.plot()

    # param = sim.parameter_values

    # Save concentrations for initial conditions

    neg_surf = "X-averaged negative particle surface concentration [mol.m-3]"
    pos_surf = "X-averaged positive particle surface concentration [mol.m-3]"
    sol_charged = sim.solution.cycles[2]
    v_upper_lim_neg = sol_charged[neg_surf].entries[-1]
    v_upper_lim_pos = sol_charged[pos_surf].entries[-1]

    # Save lower concentrations
    sol_dischg = sim.solution.cycles[3]
    v_lower_lim_neg = sol_dischg[neg_surf].entries[-1]
    v_lower_lim_pos = sol_dischg[pos_surf].entries[-1]

    # Use max conc. for normalization
    c_s_n_max = parameter_values[
        "Maximum concentration in negative electrode [mol.m-3]"
    ]
    c_s_p_max = parameter_values[
        "Maximum concentration in positive electrode [mol.m-3]"
    ]

    # These are now the min and max concs for full range of SoC
    x_n_max = v_upper_lim_neg / c_s_n_max
    x_p_min = v_upper_lim_pos / c_s_p_max
    x_n_min = v_lower_lim_neg / c_s_n_max
    x_p_max = v_lower_lim_pos / c_s_p_max

    # Work out total capacity between voltage lims

    current = sol_dischg["Current [A]"].entries
    time = sol_dischg["Time [h]"].entries
    dt = time[1:] - time[:-1]
    c_ave = (current[1:] + current[:-1]) / 2
    charge = np.cumsum(c_ave * dt)
    plt.figure()
    plt.plot(time[1:], charge)
    plt.xlabel("Time [h]")
    plt.ylabel("Cumulative Charge Transferred [Ah]")

    SoC = np.linspace(0, 1, 1000)
    x_n = x_n_min + (x_n_max - x_n_min) * SoC
    x_p = x_p_max - (x_p_max - x_p_min) * SoC

    # Ocp are functions
    U_n = parameter_values["Negative electrode OCP [V]"]
    U_p = parameter_values["Positive electrode OCP [V]"]
    U_n_eval = parameter_values.evaluate(U_n(pybamm.Array(x_n)))
    U_p_eval = parameter_values.evaluate(U_p(pybamm.Array(x_p)))
    OCV = U_p_eval - U_n_eval
    OCV = OCV.flatten()

    # Compare to C/100
    plt.figure()
    plt.plot(SoC, OCV)
    plt.ylabel("OCP [V]")
    plt.xlabel("SoC")
    terminal = sol_dischg["Terminal voltage [V]"].entries
    plt.plot(np.linspace(0.0, 1.0, len(terminal))[::-1], terminal, "r--")

    # Reverse interpolants to get back lithiation states from SoC and OCV
    x_n_SoC = interp1d(SoC, x_n)
    x_p_SoC = interp1d(SoC, x_p)
    x_n_OCV = interp1d(OCV, x_n)
    x_p_OCV = interp1d(OCV, x_p)

    vmin = float(V_lower_limit)
    vmax = float(V_upper_limit)

    init_funcs = {
        "x_n_SoC": x_n_SoC,
        "x_p_SoC": x_p_SoC,
        "x_n_OCV": x_n_OCV,
        "x_p_OCV": x_p_OCV,
        "c_s_n_max": c_s_n_max,
        "c_s_p_max": c_s_p_max,
        "vmin": vmin,
        "vmax": vmax,
    }
    with open(init_fname, "wb") as handle:
        pickle.dump(init_funcs, handle)
