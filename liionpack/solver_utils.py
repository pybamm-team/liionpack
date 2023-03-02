#
# Solver utilities
#

import casadi
import pybamm
import numpy as np
import liionpack as lp


def _serial_eval(model, solutions, inputs_dict, variables, t_eval):
    """
    Internal function to evaluate the model variables in a serial way.

    Args:
        model (pybamm.Model):
            The built model
        solutions (iter of pybamm.Solution):
            Used to get the last state of the system and use as x0 and z0 for the
            casadi integrator. Provide solution objects for each battery.
        inputs_dict (iter of input_dicts):
            Provide inputs_dict objects for each battery.
        variables (variables evaluator):
            Produced by _create_casadi_objects when mapped = False
        t_eval (np.ndarray):
            A float array of times to evaluate.
            Produced by _create_casadi_objects when mapped = False

    Returns:
        sol (list):
            solutions that have been stepped forward by one timestep
        var_eval (list):
            evaluated variables for final state of system

    """
    len_rhs = model.concatenated_rhs.size
    N = len(solutions)
    t_min = 0.0
    var_eval = []
    for k in range(N):
        if solutions[k] is None:
            # First pass
            xend = model.y0[:len_rhs]
        else:
            xend = solutions[k].y[:, -1]

        temp = inputs_dict[k]
        inputs = casadi.vertcat(*[x for x in temp.values()] + [t_min])
        ninputs = len(temp.values())
        var_eval.append(variables(0, xend[:len_rhs], xend[len_rhs:], inputs[0:ninputs]))

    return casadi.horzcat(*var_eval)


def _serial_step(model, solutions, inputs_dict, integrator, variables, t_eval, events):
    """
    Internal function to process the model for one timestep in a serial way.

    Args:
        model (pybamm.Model):
            The built model
        solutions (iter of pybamm.Solution):
            Used to get the last state of the system and use as x0 and z0 for the
            casadi integrator. Provide solution objects for each battery.
        inputs_dict (iter of input_dicts):
            Provide inputs_dict objects for each battery.
        integrator (casadi.integrator):
            Produced by _create_casadi_objects when mapped = False
        variables (variables evaluator):
            Produced by _create_casadi_objects when mapped = False
        t_eval (np.ndarray):
            A float array of times to evaluate.
            Produced by _create_casadi_objects when mapped = False
        events (mapped events evaluator):
            Produced by `_create_casadi_objects`

    Returns:
        sol (list):
            solutions that have been stepped forward by one timestep
        var_eval (list):
            evaluated variables for final state of system

    """
    len_rhs = model.concatenated_rhs.size
    N = len(solutions)
    t_min = 0.0
    timer = pybamm.Timer()
    sol = []
    var_eval = []
    events_eval = []
    for k in range(N):
        if solutions[k] is None:
            # First pass
            x0 = model.y0[:len_rhs]
            z0 = model.y0[len_rhs:]
        else:
            x0 = solutions[k].y[:len_rhs, -1]
            z0 = solutions[k].y[len_rhs:, -1]
        temp = inputs_dict[k]
        inputs = casadi.vertcat(*[x for x in temp.values()] + [t_min])
        ninputs = len(temp.values())
        # Call the integrator once, with the grid
        casadi_sol = integrator(x0=x0, z0=z0, p=inputs)
        xf = casadi_sol["xf"]
        zf = casadi_sol["zf"]
        if zf.is_empty():
            y_sol = xf
        else:
            y_sol = casadi.vertcat(xf, zf)
        xend = y_sol[:, -1]
        sol.append(pybamm.Solution(t_eval, y_sol, model, inputs_dict[k]))
        var_eval.append(variables(0, xend[:len_rhs], xend[len_rhs:], inputs[0:ninputs]))
        if events is not None:
            events_eval.append(
                events(0, xend[:len_rhs], xend[len_rhs:], inputs[0:ninputs])
            )
        integration_time = timer.time()
        sol[-1].integration_time = integration_time

    return sol, casadi.horzcat(*var_eval), casadi.horzcat(*events_eval)


def _mapped_eval(model, solutions, inputs_dict, variables, t_eval):
    """
    Internal function to evaluate the model variables in a mapped way.

    Arg:
        model (pybamm.lithium_ion.BaseModel):
            The built battery model
        solutions (iter of pybamm.Solution):
            Used to get the last state of the system and use as x0 and z0 for the
            casadi integrator. Provide solution objects for each battery.
        inputs_dict (iter of input_dicts):
            Provide inputs_dict objects for each battery.
        variables (mapped variables evaluator):
            Produced by `_create_casadi_objects`
        t_eval (np.ndarray):
            A float array of times to evaluate.
            Produced by _create_casadi_objects when mapped = False

    Returns:
        var_eval (list):
            Evaluated variables for final state of system

    """
    len_rhs = model.concatenated_rhs.size
    N = len(solutions)
    if solutions[0] is None:
        # First pass
        xend = casadi.horzcat(*[model.y0[:len_rhs] for i in range(N)])
    else:
        xend = casadi.horzcat(*[sol.y[:len_rhs, -1] for sol in solutions])
    t_min = 0.0
    inputs = []
    for temp in inputs_dict:
        inputs.append(casadi.vertcat(*[x for x in temp.values()] + [t_min]))
    ninputs = len(temp.values())
    inputs = casadi.horzcat(*inputs)
    var_eval = variables(0, xend[:len_rhs, :], xend[len_rhs:, :], inputs[0:ninputs, :])

    return var_eval


def _mapped_step(model, solutions, inputs_dict, integrator, variables, t_eval, events):
    """
    Internal function to process the model for one timestep in a mapped way.
    Mapped versions of the integrator and variables functions should already
    have been made.

    Arg:
        model (pybamm.lithium_ion.BaseModel):
            The built battery model
        solutions (iter of pybamm.Solution):
            Used to get the last state of the system and use as x0 and z0 for the
            casadi integrator. Provide solution objects for each battery.
        inputs_dict (iter of input_dicts):
            Provide inputs_dict objects for each battery.
        integrator (mapped casadi.integrator):
            Produced by `_create_casadi_objects`
        variables (mapped variables evaluator):
            Produced by `_create_casadi_objects`
        t_eval (np.ndarray):
            A float array of times to evaluate.
            Produced by _create_casadi_objects when mapped = False
        events (mapped events evaluator):
            Produced by `_create_casadi_objects`

    Returns:
        sol (list):
            Solutions that have been stepped forward by one timestep
        var_eval (list):
            Evaluated variables for final state of system

    """
    len_rhs = model.concatenated_rhs.size
    N = len(solutions)
    if solutions[0] is None:
        # First pass
        x0 = casadi.horzcat(*[model.y0[:len_rhs] for i in range(N)])
        z0 = casadi.horzcat(*[model.y0[len_rhs:] for i in range(N)])
    else:
        x0 = casadi.horzcat(*[sol.y[:len_rhs, -1] for sol in solutions])
        z0 = casadi.horzcat(*[sol.y[len_rhs:, -1] for sol in solutions])
    # t_min = [0.0]*N
    t_min = 0.0
    inputs = []
    for temp in inputs_dict:
        inputs.append(casadi.vertcat(*[x for x in temp.values()] + [t_min]))
    ninputs = len(temp.values())
    inputs = casadi.horzcat(*inputs)
    # p = casadi.horzcat(*zip(inputs, external_variables, [t_min]*N))
    # inputs_with_tmin = casadi.vertcat(inputs, np.asarray(t_min))
    # Call the integrator once, with the grid
    timer = pybamm.Timer()
    tic = timer.time()
    casadi_sol = integrator(x0=x0, z0=z0, p=inputs)
    integration_time = timer.time()
    nt = len(t_eval)
    xf = casadi_sol["xf"]
    zf = casadi_sol["zf"]
    sol = []
    xend = []
    events_eval = []
    for i in range(N):
        start = i * nt
        y_diff = xf[:, start : start + nt]
        if zf.is_empty():
            y_sol = y_diff
        else:
            y_alg = zf[:, start : start + nt]
            y_sol = casadi.vertcat(y_diff, y_alg)
        xend.append(y_sol[:, -1])
        # Not sure how to index into zf - need an example
        sol.append(pybamm.Solution(t_eval, y_sol, model, inputs_dict[i]))
        sol[-1].integration_time = integration_time
    toc = timer.time()
    lp.logger.debug(f"Mapped step completed in {toc - tic}")
    xend = casadi.horzcat(*xend)
    var_eval = variables(0, xend[:len_rhs, :], xend[len_rhs:, :], inputs[0:ninputs, :])
    if events is not None:
        events_eval = events(
            0, xend[:len_rhs, :], xend[len_rhs:, :], inputs[0:ninputs, :]
        )
    return sol, var_eval, events_eval


def _create_casadi_objects(inputs, sim, dt, Nspm, nproc, variable_names, mapped):
    """
    Internal function to produce the casadi objects in their mapped form for
    parallel evaluation

    Args:
        inputs (dict):
            initial guess for inputs (not used for simulation).
        sim (pybamm.Simulation):
            A PyBaMM simulation object that contains the model, parameter values,
            solver, solution etc.
        dt (float):
            The time interval (in seconds) for a single timestep. Fixed throughout
            the simulation
        Nspm (int):
            Number of individual batteries in the pack.
        nproc (int):
            Number of parallel processes to map to.
        variable_names (list):
            Variables to evaluate during solve. Must be a valid key in the
            model.variables
        mapped (bool):
            Use the mapped casadi objects, default is True

    Returns:
        integrator (mapped casadi.integrator):
            Solves an initial value problem (IVP) coupled to a terminal value
            problem with differential equation given as an implicit ODE coupled
            to an algebraic equation and a set of quadratures
        variables_fn (mapped variables evaluator):
            evaluates the simulation and output variables. see casadi function
        t_eval (np.ndarray):
            Float array of times to evaluate.
            times to evaluate in a single step, starting at zero for each step
        events_fn (mapped events evaluator):
            evaluates the event variables. see casadi function

    """
    solver = sim.solver
    # Initial solution - this builds the model behind the scenes
    sim.build()
    initial_solutions = []
    init_sol = sim.step(
        dt=1e-6, save=False, starting_solution=None, inputs=inputs[0]
    ).last_state
    # evaluate initial condition
    model = sim.built_model
    y0_total_size = (
        model.len_rhs + model.len_rhs_sens + model.len_alg + model.len_alg_sens
    )
    y_zero = np.zeros((y0_total_size, 1))
    for inpt in inputs:
        inputs_casadi = casadi.vertcat(*[x for x in inpt.values()])
        initial_solutions.append(init_sol.copy())
        _init = model.initial_conditions_eval(0, y_zero, inputs_casadi)
        initial_solutions[-1].y[:] = _init

    # Step model forward dt seconds
    t_eval = np.linspace(0, dt, 11)

    # No external variables - Temperature solved as lumped model in pybamm
    # External variables could (and should) be used if battery thermal problem
    # Includes conduction with any other circuits or neighboring batteries
    # inp_and_ext.update(external_variables)
    inp_and_ext = inputs

    # Code to create mapped integrator
    integrator = solver.create_integrator(
        sim.built_model, inputs=inp_and_ext, t_eval=t_eval
    )
    if mapped:
        integrator = integrator.map(Nspm, "thread", nproc)
    # Get the input parameter order
    ip_order = inputs[0].keys()
    # Variables function for parallel evaluation
    casadi_objs = sim.built_model.export_casadi_objects(
        variable_names=variable_names, input_parameter_order=ip_order
    )
    variables = casadi_objs["variables"]
    t, x, z, p = (
        casadi_objs["t"],
        casadi_objs["x"],
        casadi_objs["z"],
        casadi_objs["inputs"],
    )
    variables_stacked = casadi.vertcat(*variables.values())
    variables_fn = casadi.Function("variables", [t, x, z, p], [variables_stacked])
    if mapped:
        variables_fn = variables_fn.map(Nspm, "thread", nproc)

    # Look for events in model variables and create a function to evaluate them
    all_vars = sorted(sim.model.variables.keys())
    event_vars = [v for v in all_vars if "Event" in v]
    if len(event_vars) > 0:
        # Variables function for parallel evaluation
        casadi_objs = sim.built_model.export_casadi_objects(
            variable_names=variable_names, input_parameter_order=ip_order
        )
        events = casadi_objs["variables"]
        t, x, z, p = (
            casadi_objs["t"],
            casadi_objs["x"],
            casadi_objs["z"],
            casadi_objs["inputs"],
        )
        events_stacked = casadi.vertcat(*events.values())
        events_fn = casadi.Function("variables", [t, x, z, p], [events_stacked])
        if mapped:
            events_fn = events_fn.map(Nspm, "thread", nproc)
    else:
        events_fn = None

    output = {
        "integrator": integrator,
        "variables_fn": variables_fn,
        "t_eval": t_eval,
        "event_names": event_vars,
        "events_fn": events_fn,
        "initial_solutions": initial_solutions,
    }
    return output


def solve(
    netlist=None,
    sim_func=None,
    parameter_values=None,
    experiment=None,
    inputs=None,
    initial_soc=None,
    nproc=1,
    output_variables=None,
    manager="casadi",
):
    """
    Solves a pack simulation

    Args:
        netlist (pandas.DataFrame):
            A netlist of circuit elements with format. desc, node1, node2, value.
            Produced by liionpack.read_netlist or liionpack.setup_circuit
        sim_func (function):
            A function containing model and solver definitions that accepts
            parameter_values and returns a simulation.
        parameter_values (pybamm.ParameterValues):
            A dictionary of all the model parameters
        experiment (pybamm.Experiment):
            The experiment to be simulated. experiment.period is used to
            determine the length of each timestep.
        inputs (dict):
            Dictionary for every model input with value for each battery
        initial_soc (float):
            The initial state of charge for every battery. The default is None
            in which case concentrations set in the parameter_values are used.
        nproc (int):
            Number of processes to start in parallel for mapping. The default is 1.
        output_variables (list):
            Variables to evaluate during solve. Must be a valid key in the
            model.variables
        manager (string, can be - ["casadi", "ray"]):
            The solver manager to use for solving the electrochemical problem.

    Returns:
        output (dict):
            simulation output with keys including those specified in output
            variables, values are arrays of shape - [# steps, # batteries])

    """

    if netlist is None or parameter_values is None or experiment is None:
        raise Exception("Please supply a netlist, paramater_values, and experiment")

    if manager == "casadi":
        rm = lp.CasadiManager()
    elif manager == "ray":
        rm = lp.RayManager()
    else:
        rm = lp.CasadiManager()
        lp.logger.notice("manager instruction not supported, using default")

    output = rm.solve(
        netlist=netlist,
        sim_func=sim_func,
        parameter_values=parameter_values,
        experiment=experiment,
        output_variables=output_variables,
        inputs=inputs,
        nproc=nproc,
        initial_soc=initial_soc,
        setup_only=False,
    )
    return output
