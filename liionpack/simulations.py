# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:37:51 2021

@author: tom
"""

import pybamm


def _current_function(t):
    r"""
    Internal function to make current an input parameter

    Parameters
    ----------
    t : float
        Dummy time parameter.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return pybamm.InputParameter("Current")


def create_simulation(parameter_values=None, experiment=None, make_inputs=False):
    r"""
    Create a PyBaMM simulation set up for interation with liionpack

    Parameters
    ----------
    parameter_values : pybamm.ParameterValues class
        DESCRIPTION. The default is None.
    experiment : pybamm.Experiment class
        DESCRIPTION. The default is None.
    make_inputs : bool, optional
        Changes "Current function [A]" and "Total heat transfer coefficient
        [W.m-2.K-1]" to be inputs that are controlled by liionpack.
        The default is False.

    Returns
    -------
    sim : pybamm.Simulation
        A simulation that can be solved individually or passed into the
        liionpack solve method

    """
    # Create the pybamm model
    model = pybamm.lithium_ion.SPMe(
        options={
            "thermal": "lumped",
        }
    )
    # geometry = model.default_geometry
    if parameter_values is None:
        # load parameter values and process model and geometry
        chemistry = pybamm.parameter_sets.Chen2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
    # Change the current function to be an input as this is set by the external circuit
    if make_inputs:
        parameter_values.update(
            {
                "Current function [A]": _current_function,
            }
        )
        parameter_values.update(
            {
                "Current": "[input]",
                "Total heat transfer coefficient [W.m-2.K-1]": "[input]",
            },
            check_already_exists=False,
        )

    solver = pybamm.CasadiSolver(mode="safe")
    sim = pybamm.Simulation(
        model=model,
        experiment=experiment,
        parameter_values=parameter_values,
        solver=solver,
    )
    return sim


if __name__ == "__main__":
    sim = create_simulation()
    sim.solve([0, 1800])
    sim.plot()
