#
# Simulation utilities
#
import pickle
import pybamm
import liionpack as lp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

init_fname = os.path.join(lp.INIT_FUNCS, "init_funcs.pickle")


def update_init_conc(sim, SoC=-1, OCV=-1, method="calculation"):
    """
    Update initial concentration parameters

    Parameters
    ----------
    sim : :class:`pybamm.Simulation`
        The battery simulation.
    SoC : float, optional
        Target initial SoC. Must be between 0 and 1. Default is -1, in which
        case the initial concentrations are set using the target OCV.
    OCV : float, optional
        Target initial OCV. Must be between 0 and 1. Default is -1, in which
        case the initial concentrations are set using the target SoC. This option is
        only used if method is "experiment".
    method : str, optional
        The method used to compute the initial concentrations. Can be "calculation",
        in which case `pybamm.get_initial_stoichiometries` is used to compute initial
        stoichiometries that give the desired initial state of charge, or "experiment",
        in which case a slow discharge between voltage limits is used to compute the initial
        concentrations given a target SoC or OCV. Default is "calculation".
    """
    param = sim.parameter_values

    if method == "calculation":
        c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]
        c_p_max = param["Maximum concentration in positive electrode [mol.m-3]"]
        x, y = pybamm.lithium_ion.get_initial_stoichiometries(SoC, param)
        c_s_n_init, c_s_p_init = x * c_n_max, y * c_p_max
    elif method == "experiment":
        c_s_n_init, c_s_p_init = initial_conditions_from_experiment(
            param, SoC=SoC, OCV=OCV
        )
    else:
        raise ValueError("'method' must be 'calculation' or 'experiment'.")

    param.update(
        {
            "Initial concentration in negative electrode [mol.m-3]": c_s_n_init,
            "Initial concentration in positive electrode [mol.m-3]": c_s_p_init,
        }
    )


def initial_conditions_from_experiment(parameter_values, SoC=-1, OCV=-1):
    """
    Update initial concentration parameters using an experiment between voltage limits.


    Parameters
    ----------
    parameter_values : :class:`pybamm.ParamaterValues`
        The parameter values used in the simulation.
    SoC : float, optional
        Target initial SoC. Must be between 0 and 1. Default is -1, in which
        case the initial concentrations are set using the target OCV.
    OCV : float, optional
        Target initial OCV. Must be between 0 and 1. Default is -1, in which
        case the initial concentrations are set using the target SoC. This option is
        only used if method is "experiment".

    Returns
    -------
    c_s_n_init : float
        The initial concentration in the negative electrode.
    c_s_p_init : float
        The initial concentration in the positive electrode.
    """

    # TODO: what if someone changes parameters? We should make this every time,
    # but I see why you want to pickle it for speed.
    try:
        with open(init_fname, "rb") as handle:
            init_funcs = pickle.load(handle)
    except FileNotFoundError:
        init_funcs = create_init_funcs(parameter_values)
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


def create_init_funcs(parameter_values):
    """
    Run an experiment which can be used to create interpolants to determine the
    initial concentrations corresponding to a given initial state of charge or
    open circuit voltage.

    Parameters
    ----------
    parameter_values : :class:`pybamm.ParamaterValues`
        The parameter values used in the simulation.
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

    # Set up and solve simulation
    sim = lp.create_simulation(parameter_values=parameter_values, experiment=experiment)
    sim.solve()

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

    # Use max concentration for normalization
    c_s_n_max = parameter_values[
        "Maximum concentration in negative electrode [mol.m-3]"
    ]
    c_s_p_max = parameter_values[
        "Maximum concentration in positive electrode [mol.m-3]"
    ]

    # These are now the min and max concentrations for full range of SoC
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

    # OCP are functions
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
