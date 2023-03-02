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
        param = pybamm.ParameterValues("Chen2020")
    else:
        param = parameter_values.copy()

    # Set up solver and simulation
    solver = pybamm.CasadiSolver(mode="safe")
    sim = pybamm.Simulation(
        model=model,
        parameter_values=param,
        solver=solver,
    )
    return sim


def thermal_simulation(parameter_values=None):
    """
    Create a PyBaMM simulation set up for integration with liionpack

    Args:
        parameter_values (pybamm.ParameterValues):
            The default is None.

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
        parameter_values = pybamm.ParameterValues("Chen2020")

    # Change the heat transfer coefficient to be an input controlled by the
    # external circuit
    parameter_values.update(
        {
            "Total heat transfer coefficient [W.m-2.K-1]": "[input]",
        },
    )

    # Set up solver and simulation
    solver = pybamm.CasadiSolver(mode="safe")
    sim = pybamm.Simulation(
        model=model,
        parameter_values=parameter_values,
        solver=solver,
    )
    return sim


def thermal_external(parameter_values=None):
    """
    Create a PyBaMM simulation set up for integration with liionpack.
    External thermal option is used so that temperature dependence can be
    included in models but temperature supplied by another algorithm. This
    is useful for packs and cells where thermal connections are seperate or
    distinct from electrical connections.

    Args:
        parameter_values (pybamm.ParameterValues):
            The default is None.

    Returns:
        pybamm.Simulation:
            A simulation that can be solved individually or passed into the
            liionpack solve method

    """
    # Create the pybamm model
    model = pybamm.lithium_ion.SPMe(
        options={
            "calculate heat source for isothermal models": "true",
            "cell geometry": "arbitrary",
            "dimensionality": 0,
            "thermal": "isothermal",
        }
    )

    # Add events to the model
    model = lp.add_events_to_model(model)

    # Set up parameter values
    if parameter_values is None:
        parameter_values = pybamm.ParameterValues("Chen2020")

    # Change the ambient temperature to be an input controlled by the
    # external circuit
    parameter_values["Ambient temperature [K]"] = pybamm.InputParameter(
        "Input temperature [K]"
    )
    parameter_values["Initial temperature [K]"] = pybamm.InputParameter(
        "Input temperature [K]"
    )

    # Set up solver and simulation
    solver = pybamm.CasadiSolver(mode="safe")
    sim = pybamm.Simulation(
        model=model,
        parameter_values=parameter_values,
        solver=solver,
    )
    return sim
