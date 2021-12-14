#
# Simulation utilities
#
import pybamm
import liionpack as lp
import numpy as np


def get_initial_stoichiometries(initial_soc, parameter_values):
    """
    Calculate initial stoichiometries to start off the simulation at a particular
    state of charge, given voltage limits, open-circuit potentials, etc defined by
    parameter_values

    Args:
        initial_soc (float):
            Target initial SOC. Must be between 0 and 1.
        parameter_values (pybamm.ParameterValues):
            The parameter values class that will be used for the simulation.
            Required for calculating appropriate initial stoichiometries.

    Returns:
        x, y (float):
            The initial stoichiometries that give the desired initial state of charge
    """
    if np.any(initial_soc < 0) or np.any(initial_soc > 1):
        raise ValueError("Initial SOC should be between 0 and 1")

    model = pybamm.lithium_ion.ElectrodeSOH()

    param = pybamm.LithiumIonParameters()
    sim = pybamm.Simulation(model, parameter_values=parameter_values)

    V_min = parameter_values.evaluate(param.voltage_low_cut_dimensional)
    V_max = parameter_values.evaluate(param.voltage_high_cut_dimensional)
    C_n = parameter_values.evaluate(param.C_n_init)
    C_p = parameter_values.evaluate(param.C_p_init)
    n_Li = parameter_values.evaluate(param.n_Li_particles_init)

    # Solve the model and check outputs
    sol = sim.solve(
        [0],
        inputs={
            "V_min": V_min,
            "V_max": V_max,
            "C_n": C_n,
            "C_p": C_p,
            "n_Li": n_Li,
        },
    )

    x_0 = sol["x_0"].data[0]
    y_0 = sol["y_0"].data[0]
    C = sol["C"].data[0]
    x = x_0 + np.asarray(initial_soc) * C / C_n
    y = y_0 - np.asarray(initial_soc) * C / C_p

    return x, y


def update_init_conc(param, SoC=None, update=True):
    """
    Update initial concentration parameters

    Args:
        param (pybamm.ParameterValues):
            The battery simulation parameters.
        SoC (float):
            Target initial SoC. Must be between 0 and 1. Default is -1, in which
            case the initial concentrations are set using the target OCV.

    Returns:
        c_s_n_init, c_s_p_init (float):
            initial concentrations in negative and positive particles
    """
    c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    c_p_max = param["Maximum concentration in positive electrode [mol.m-3]"]
    x, y = lp.get_initial_stoichiometries(SoC, param)
    c_s_n_init, c_s_p_init = x * c_n_max, y * c_p_max
    if update:
        param.update(
            {
                "Initial concentration in negative electrode [mol.m-3]": c_s_n_init,
                "Initial concentration in positive electrode [mol.m-3]": c_s_p_init,
            }
        )
    return c_s_n_init, c_s_p_init
