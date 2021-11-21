#
# Simulation
#

import pybamm


def create_simulation(parameter_values=None, experiment=None, make_inputs=False):
    """
    Create a PyBaMM simulation set up for interation with liionpack

    Args:
        parameter_values (pybamm.ParameterValues):
            The default is None.
        experiment (pybamm.Experiment):
            The default is None.
        make_inputs (bool):
            Changes "Current function [A]" and "Total heat transfer coefficient
            [W.m-2.K-1]" to be inputs that are controlled by liionpack.
            The default is False.

    Returns:
        pybamm.Simulation:
            A simulation that can be solved individually or passed into the
            liionpack solve method

    """
    # Create the pybamm model
    model = pybamm.lithium_ion.SPMe(
        options={
            "thermal": "lumped",
        }
    )

    # Set up parameter values
    if parameter_values is None:
        chemistry = pybamm.parameter_sets.Chen2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)

    # Change the current function and heat transfer coefficient to be
    # inputs controlled by the external circuit
    if make_inputs:
        parameter_values.update(
            {
                "Current function [A]": "[input]",
                "Total heat transfer coefficient [W.m-2.K-1]": "[input]",
            },
        )

    # Set up solver and simulation
    solver = pybamm.CasadiSolver(mode="safe")
    sim = pybamm.Simulation(
        model=model,
        experiment=experiment,
        parameter_values=parameter_values,
        solver=solver,
    )

    return sim
