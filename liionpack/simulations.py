#
# Simulation
#

import pybamm
import liionpack as lp

tscale = 10


def _replace_timescale(model, parameter_values):
    symbolic_tau = pybamm.LithiumIonParameters().tau_discharge
    tau = parameter_values.process_symbol(symbolic_tau)
    try:
        tau_eval = tau.evaluate()
        model.timescale = pybamm.Scalar(tau_eval)
    except KeyError:
        # A child of the timescale is an input
        keys = [
            "Maximum concentration in negative electrode [mol.m-3]",
            "Negative electrode thickness [m]",
            "Separator thickness [m]",
            "Positive electrode thickness [m]",
            "Typical current [A]",
            "Number of electrodes connected in parallel to make a cell",
            "Electrode width [m]",
            "Electrode height [m]",
        ]
        for key in keys:
            if parameter_values[key].__class__ is pybamm.InputParameter:
                lp.logger.warn(
                    key,
                    "is an input parameter that affects the "
                    + "timescale, setting timescale to typical timescale",
                )
        model.timescale = pybamm.Scalar(parameter_values["Typical timescale [s]"])


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
    model = pybamm.lithium_ion.SPM({"timescale": tscale})

    # Add events to the model
    model = lp.add_events_to_model(model)

    # Set up parameter values
    if parameter_values is None:
        chemistry = pybamm.parameter_sets.Chen2020
        param = pybamm.ParameterValues(chemistry=chemistry)
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
            "timescale": tscale,
        }
    )

    # Add events to the model
    model = lp.add_events_to_model(model)

    # Set up parameter values
    if parameter_values is None:
        chemistry = pybamm.parameter_sets.Chen2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)

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
            "thermal": "lumped",
            "external submodels": ["thermal"],
            "timescale": tscale,
        }
    )

    # Add events to the model
    model = lp.add_events_to_model(model)

    # Set up parameter values
    if parameter_values is None:
        chemistry = pybamm.parameter_sets.Chen2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)

    # _replace_timescale(model, parameter_values)
    # Change the current function and heat transfer coefficient to be
    # inputs controlled by the external circuit
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
