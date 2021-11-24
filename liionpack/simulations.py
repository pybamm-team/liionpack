#
# Simulation
#

import pybamm
import liionpack as lp


def basic_simulation(parameter_values=None):
    """
    Create a Basic PyBaMM simulation set up for integration with liionpack

    Args:
        parameter_values (pybamm.ParameterValues):
            The default is None.

    Returns:
        pybamm.Simulation:
            A simulation that can be solved individually or passed into the
            liionpack solve method

    """
    # Create the pybamm model
    model = pybamm.lithium_ion.SPM()

    # Add events to the model
    model = lp.add_events_to_model(model)

    # Set up parameter values
    if parameter_values is None:
        chemistry = pybamm.parameter_sets.Chen2020
        param = pybamm.ParameterValues(chemistry=chemistry)
    else:
        param = parameter_values.copy()

    # Change the current function to be input controlled by the external circuit
    param.update(
        {
            "Current function [A]": "[input]",
        },
    )

    # Set up solver and simulation
    solver = pybamm.CasadiSolver(mode="safe")
    sim = pybamm.Simulation(
        model=model,
        experiment=None,
        parameter_values=param,
        solver=solver,
    )

    return sim


def create_simulation(parameter_values=None, experiment=None, make_inputs=False):
    """
    Create a PyBaMM simulation set up for integration with liionpack

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

    # Add events to the model
    model = lp.add_events_to_model(model)

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
