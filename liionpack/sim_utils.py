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

    param = pybamm.LithiumIonParameters()
    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)
    return esoh_solver.get_initial_stoichiometries(initial_soc)


def update_init_conc(param, SoC=None, update=True):
    """
    Update initial concentration parameters

    Args:
        param (pybamm.ParameterValues):
            The battery simulation parameters.
        SoC (float):
            Target initial SoC. Must be between 0 and 1. Default is -1, in which
            case the initial concentrations are set using the target OCV.
        update (bool):
            Update the initial concentrations in place if True

    Returns:
        c_s_n_init, c_s_p_init (float):
            initial concentrations in negative and positive particles
    """
    c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    c_p_max = param["Maximum concentration in positive electrode [mol.m-3]"]
    x, y = lp.get_initial_stoichiometries(SoC, param)
    if x is not None:
        c_s_n_init, c_s_p_init = x * c_n_max, y * c_p_max
    else:
        return x, y
    if update:
        param.update(
            {
                "Initial concentration in negative electrode [mol.m-3]": c_s_n_init,
                "Initial concentration in positive electrode [mol.m-3]": c_s_p_init,
            }
        )
    return c_s_n_init, c_s_p_init
