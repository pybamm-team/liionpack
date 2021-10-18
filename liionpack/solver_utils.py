# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:44:31 2021

@author: Tom
"""
import casadi
import pybamm
import numpy as np
import time as ticker
import liionpack as lp
import matplotlib.pyplot as plt
from tqdm import tqdm
from dask.distributed import Client


def _serial_step(model, solutions, inputs_dict, integrator, variables, t_eval):
    r'''
    Internal function to process the model for one timestep in a serial way.

    Parameters
    ----------
    model : pybamm.Model
        The built model
    solutions : list of pybamm.Solution objects for each battery
        Used to get the last state of the system and use as x0 and z0 for the
        casadi integrator
    inputs_dict : list of inputs_dict objects for each battery
        DESCRIPTION.
    integrator : casadi.integrator
        Produced by _create_casadi_objects when mapped = False
    variables : variables evaluator
        Produced by _create_casadi_objects when mapped = False
    t_eval : float array of times to evaluate
        Produced by _create_casadi_objects when mapped = False

    Returns
    -------
    sol : list
        solutions that have been stepped forward by one timestep
    var_eval : list
        evaluated variables for final state of system

    '''
    len_rhs = model.concatenated_rhs.size
    N = len(solutions)
    t_min = 0.0
    timer = pybamm.Timer()
    sol = []
    var_eval = []
    for k in range(N):
        if solutions[k] is None:
            # First pass
            x0 = model.y0[:len_rhs]
            z0 = model.y0[len_rhs:]
        else:
            x0 = solutions[k].y[:len_rhs, -1]
            z0 = solutions[k].y[len_rhs:, -1]
        temp = inputs_dict[k]
        inputs = casadi.vertcat(*[x for x in temp.values()]+[t_min])
        ninputs = len(temp.values())
        # Call the integrator once, with the grid
        casadi_sol = integrator(
            x0=x0, z0=z0, p=inputs
        )
        y_sol = casadi_sol["xf"]
        xend = y_sol[:, -1]
        sol.append(pybamm.Solution(t_eval, y_sol, model,
                                   inputs_dict[k]))
        var_eval.append(variables(0, xend, 0, inputs[0:ninputs]))
        integration_time = timer.time()
        sol[-1].integration_time = integration_time

    return sol, casadi.horzcat(*var_eval)

def _mapped_step(model, solutions, inputs_dict, integrator, variables, t_eval):
    r'''
    Internal function to process the model for one timestep in a mapped way.
    Mapped versions of the integrator and variables functions should already
    have been made.

    Parameters
    ----------
    model : pybamm.Model
        The built model
    solutions : list of pybamm.Solution objects for each battery
        Used to get the last state of the system and use as x0 and z0 for the
        casadi integrator
    inputs_dict : list of inputs_dict objects for each battery
        DESCRIPTION.
    integrator : mapped casadi.integrator
        Produced by _create_casadi_objects when mapped = True
    variables : mapped variables evaluator
        Produced by _create_casadi_objects when mapped = True
    t_eval : float array of times to evaluate
        Produced by _create_casadi_objects when mapped = True

    Returns
    -------
    sol : list
        solutions that have been stepped forward by one timestep
    var_eval : list
        evaluated variables for final state of system

    '''
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
        inputs.append(casadi.vertcat(*[x for x in temp.values()]+[t_min]))
    ninputs = len(temp.values())
    inputs = casadi.horzcat(*inputs)
    # p = casadi.horzcat(*zip(inputs, external_variables, [t_min]*N))
    # inputs_with_tmin = casadi.vertcat(inputs, np.asarray(t_min))
    # Call the integrator once, with the grid
    timer = pybamm.Timer()
    casadi_sol = integrator(
        x0=x0, z0=z0, p=inputs
    )
    integration_time = timer.time()
    nt = len(t_eval)
    xf = casadi_sol["xf"]
    # zf = casadi_sol["zf"]
    sol = []
    xend = []
    for i in range(N):
        start = i * nt
        y_sol = xf[:, start:start + nt]
        xend.append(y_sol[:, -1])
        # Not sure how to index into zf - need an example
        sol.append(pybamm.Solution(t_eval, y_sol, model,
                                   inputs_dict[i]))
        sol[-1].integration_time = integration_time
    xend = casadi.horzcat(*xend)
    var_eval = variables(0, xend, 0, inputs[0:ninputs, :])
    return sol, var_eval

def _create_casadi_objects(I_init, htc, sim, dt, Nspm, nproc, variable_names,
                           mapped):
    r'''
    Internal function to produce the casadi objects in their mapped form for
    parallel evaluation

    Parameters
    ----------
    I_init : float
        initial guess for current of a battery (not used for simulation).
    htc : float
        initial guess for htc of a battery (not used for simulation).
    sim : pybamm.Simulation
        A PyBaMM simulation object that contains the model, parameter_values,
        solver, solution etc.
    dt : float
        The time interval for a single timestep. Fixed throughout the simulation
    Nspm : int
        Number of individual batteries in the pack.
    nproc : int
        Number of parallel processes to map to.
    variable_names : list
        Variables to evaluate during solve. Must be a valid key in the
        model.variables
    mapped : boolean
        Use the mapped casadi objects, default is True

    Returns
    -------
    integrator : mapped casadi.integrator
        Solves an initial value problem (IVP) coupled to a terminal value
        problem with differential equation given as an implicit ODE coupled
        to an algebraic equation and a set of quadratures
    variables_fn : mapped variables evaluator
        evaluates the simulation and output variables. see casadi function
    t_eval : float array of times to evaluate
        times to evaluate in a single step, starting at zero for each step

    '''
    inputs = {"Current": I_init,
              "Total heat transfer coefficient [W.m-2.K-1]": htc}
    solver = sim.solver
    # solve model for 1 second to initialise the circuit
    t_eval = np.linspace(0, 1, 2)
    # Initial solution - this builds the model behind the scenes
    sol_init = sim.solve(t_eval, inputs=inputs)
    # step model
    # Code to create mapped integrator
    t_eval = np.linspace(0, dt, 11)
    t_eval_ndim = t_eval / sim.model.timescale.evaluate()
    inp_and_ext = inputs
    # No external variables - Temperature solved as lumped model in pybamm
    # External variables could (and should) be used if battery thermal problem
    # Includes conduction with any other circuits or neighboring batteries
    # inp_and_ext.update(external_variables)
    
    integrator = solver.create_integrator(sim.built_model, inputs=inp_and_ext,
                                          t_eval=t_eval_ndim)
    if mapped:
        integrator = integrator.map(Nspm, 'thread', nproc)

    # Variables function for parallel evaluation
    casadi_objs = sim.built_model.export_casadi_objects(variable_names=variable_names)
    variables = casadi_objs['variables']
    t, x, z, p = casadi_objs["t"], casadi_objs["x"], casadi_objs["z"], casadi_objs["inputs"]
    variables_stacked = casadi.vertcat(*variables.values())
    variables_fn = casadi.Function("variables", [t, x, z, p], [variables_stacked])
    if mapped:
        variables_fn = variables_fn.map(Nspm, 'thread', nproc)
    return integrator, variables_fn, t_eval


def solve(netlist=None, parameter_values=None, experiment=None,
          I_init=1.0, htc=None, initial_soc=0.5,
          nproc=12,
          output_variables=None,
          mapped=True):
    r'''
    Solves a pack simulation

    Parameters
    ----------
    netlist : pandas.DataFrame
        A netlist of circuit elements with format. desc, node1, node2, value.
        Produced by liionpack.read_netlist or liionpack.setup_circuit
    parameter_values : pybamm.ParameterValues class
        A dictionary of all the model parameters
    experiment : pybamm.Experiment class
        The experiment to be simulated. experiment.period is used to
        determine the length of each timestep.
    I_init : float, optional
        Initial guess for single battery current [A]. The default is 1.0.
    htc : float array, optional
        Heat transfer coefficient array of length Nspm. The default is None.
    initial_soc : float
        The initial state of charge for every battery. The default is 0.5
    nproc : int, optional
        Number of processes to start in parallel for mapping. The default is 12.
    output_variables : list, optional
        Variables to evaluate during solve. Must be a valid key in the
        model.variables
    mapped : boolean
        Use the mapped casadi objects, default is True

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    output : ndarray shape [# variable, # steps, # batteries]
        simulation output array

    '''

    if netlist is None or parameter_values is None or experiment is None:
        raise Exception('Please supply a netlist, paramater_values, and experiment')

    # Get netlist indices for resistors, voltage sources, current sources
    Ri_map = netlist['desc'].str.find('Ri') > -1
    V_map = netlist['desc'].str.find('V') > -1
    I_map = netlist['desc'].str.find('I') > -1
    
    Nspm = np.sum(V_map)

    protocol = lp.generate_protocol_from_experiment(experiment)
    dt = experiment.period
    Nsteps = len(protocol)
    
    # Solve the circuit to initialise the electrochemical models
    V_node, I_batt = lp.solve_circuit(netlist)

    sim = lp.create_simulation(parameter_values, make_inputs=True)
    lp.update_init_conc(sim, SoC=initial_soc)

    v_cut_lower = parameter_values['Lower voltage cut-off [V]']
    v_cut_higher = parameter_values['Upper voltage cut-off [V]']

    # The simulation output variables calculated at each step for each battery
    # Must be a 0D variable i.e. battery wide volume average - or X-averaged for 1D model
    variable_names = ['Terminal voltage [V]',
                      'Measured battery open circuit voltage [V]',
                      'Local ECM resistance [Ohm]']
    if output_variables is not None:
        for out in output_variables:
            if out not in variable_names:
                variable_names.append(out)
        # variable_names = variable_names + output_variables
    Nvar = len(variable_names)
    # Storage variables for simulation data
    shm_i_app = np.zeros([Nsteps, Nspm], dtype=float)
    shm_Ri = np.zeros([Nsteps, Nspm], dtype=float)
    output = np.zeros([Nvar, Nsteps, Nspm], dtype=float)

    # Initialize currents in battery models
    shm_i_app[0, :] = I_batt * -1


    time = 0
    # step = 0
    end_time = dt*Nsteps
    step_solutions = [None] * Nspm
    V_terminal = []
    record_times = []

   
    integrator, variables_fn, t_eval = _create_casadi_objects(I_init, htc[0],
                                                              sim, dt, Nspm,
                                                              nproc,
                                                              variable_names,
                                                              mapped)

    if mapped:
        step_fn = _mapped_step
    else:
        step_fn = _serial_step
    sim_start_time = ticker.time()
    for step in tqdm(range(Nsteps), desc='Solving Pack'):
        step_solutions, var_eval = step_fn(sim.built_model, step_solutions,
                                           lp.build_inputs_dict(shm_i_app[step, :], htc),
                                           integrator, variables_fn, t_eval)
        output[:, step, :] = var_eval

        time += dt
        # Calculate internal resistance and update netlist
        temp_v = output[0, step, :]
        temp_ocv = output[1, step, :]
        temp_Ri = np.abs(output[2, step, :])
        shm_Ri[step, :] = temp_Ri


        netlist.loc[V_map, ('value')] = temp_ocv
        netlist.loc[Ri_map, ('value')] = temp_Ri
        netlist.loc[I_map, ('value')] = protocol[step]

        # print('Stepping time', np.around(ticker.time()-tic, 2), 's')
        if np.any(temp_v < v_cut_lower):
            print('Low V limit reached')
            break
        if np.any(temp_v > v_cut_higher):
            print('High V limit reached')
            break
        # step += 1
        if time <= end_time:
            record_times.append(time)
            V_node, I_batt = lp.solve_circuit(netlist)
            V_terminal.append(V_node.max())
        if time < end_time:
            shm_i_app[step+1, :] = I_batt[:] * -1
    all_output = {}
    all_output['Time [s]'] = np.asarray(record_times)
    all_output['Pack current [A]'] = np.asarray(protocol[:step+1])
    all_output['Pack terminal voltage [V]'] = np.asarray(V_terminal)
    all_output['Cell current [A]'] = shm_i_app[:step+1, :]
    for j in range(Nvar):
        all_output[variable_names[j]] = output[j, :step+1, :]
    
    toc = ticker.time()
    pybamm.logger.notice('Solve circuit time '+
                          str(np.around(toc-sim_start_time, 3)) + 's')
    return all_output


def _step_dask(input_dict, solution, model, integrator, variables, t_eval):
    """
    Calculate solution and variables for battery cell model.
    """
    len_rhs = model.concatenated_rhs.size
    t_min = 0.0
    timer = pybamm.Timer()

    if solution is None:
        # First pass
        x0 = model.y0[:len_rhs]
        z0 = model.y0[len_rhs:]
    else:
        x0 = solution.y[:len_rhs, -1]
        z0 = solution.y[len_rhs:, -1]

    inputs = casadi.vertcat(*[x for x in input_dict.values()]+[t_min])
    ninputs = len(input_dict.values())

    # Call the integrator once, with the grid
    casadi_sol = integrator(x0=x0, z0=z0, p=inputs)
    y_sol = casadi_sol["xf"]
    xend = y_sol[:, -1]

    sol = pybamm.Solution(t_eval, y_sol, model, input_dict)
    var_eval = variables(0, xend, 0, inputs[0:ninputs])

    integration_time = timer.time()
    sol.integration_time = integration_time

    results = sol, var_eval
    return results


def solve_dask(
        netlist=None, parameter_values=None, experiment=None, I_init=1.0,
        htc=None, initial_soc=0.5, nproc=12, output_variables=None, mapped=True):
    """
    Solves a battery pack simulation where Dask is used to solve the PyBaMM
    battery cell models in parallel.
    """
    if netlist is None or parameter_values is None or experiment is None:
        raise Exception('Please supply a netlist, paramater_values, and experiment')

    # Setup client for Dask distributed scheduler
    client = Client()
    print(client)
    print(client.dashboard_link)

    # Get netlist indices for resistors, voltage sources, current sources
    Ri_map = netlist['desc'].str.find('Ri') > -1
    V_map = netlist['desc'].str.find('V') > -1
    I_map = netlist['desc'].str.find('I') > -1

    # Number of battery cell models
    Nspm = sum(V_map)

    # Number of time steps from experiment
    protocol = lp.generate_protocol_from_experiment(experiment)
    dt = experiment.period
    Nsteps = len(protocol)

    # Solve the circuit to initialize the electrochemical cell models
    V_node, I_batt = lp.solve_circuit(netlist)

    # Create the simulation object with initial conditions
    sim = lp.create_simulation(parameter_values, make_inputs=True)
    lp.update_init_conc(sim, SoC=initial_soc)

    # Get cut-off voltages
    v_cut_lower = parameter_values['Lower voltage cut-off [V]']
    v_cut_higher = parameter_values['Upper voltage cut-off [V]']

    # Simulation output variables calculated at each step for each battery.
    # Must be a 0D variable i.e. battery wide volume average or X-averaged
    # for 1D model.
    variable_names = ['Terminal voltage [V]',
                      'Measured battery open circuit voltage [V]',
                      'Local ECM resistance [Ohm]']

    if output_variables is not None:
        for out in output_variables:
            if out not in variable_names:
                variable_names.append(out)

    # Number of variables associated with the simulation
    Nvar = len(variable_names)

    # Storage variables for simulation data
    shm_i_app = np.zeros([Nsteps, Nspm], dtype=float)
    shm_Ri = np.zeros([Nsteps, Nspm], dtype=float)
    output = np.zeros([Nvar, Nsteps, Nspm], dtype=float)

    # Initialize currents in battery models
    shm_i_app[0, :] = I_batt * -1

    # Initialize other variables
    time = 0
    end_time = dt * Nsteps
    step_solutions = [None] * Nspm
    V_terminal = []
    record_times = []

    # Create Casadi objects
    integrator, variables_fn, t_eval = _create_casadi_objects(
        I_init, htc[0], sim, dt, Nspm, nproc, variable_names, mapped=False)

    # Calculate solutions for each time step
    for step in tqdm(range(Nsteps), desc='Solving Pack'):

        inputs_dict = lp.build_inputs_dict(shm_i_app[step, :], htc)

        # Calculate solution for each battery cell model
        lazy_results = client.map(
            _step_dask, inputs_dict, step_solutions,
            model=sim.built_model, integrator=integrator, variables=variables_fn, t_eval=t_eval)

        results = client.gather(lazy_results)
        step_solutions, var_eval = zip(*results)

        output[:, step, :] = casadi.horzcat(*var_eval)
        time += dt

        # Calculate internal resistance and update netlist
        temp_v = output[0, step, :]
        temp_ocv = output[1, step, :]
        temp_Ri = np.abs(output[2, step, :])
        shm_Ri[step, :] = temp_Ri

        netlist.loc[V_map, ('value')] = temp_ocv
        netlist.loc[Ri_map, ('value')] = temp_Ri
        netlist.loc[I_map, ('value')] = protocol[step]

        if np.any(temp_v < v_cut_lower):
            print('Low V limit reached')
            break
        if np.any(temp_v > v_cut_higher):
            print('High V limit reached')
            break
        # step += 1
        if time <= end_time:
            record_times.append(time)
            V_node, I_batt = lp.solve_circuit(netlist)
            V_terminal.append(V_node.max())
        if time < end_time:
            shm_i_app[step+1, :] = I_batt[:] * -1

    all_output = {}
    all_output['Time [s]'] = np.asarray(record_times)
    all_output['Pack current [A]'] = np.asarray(protocol[:step+1])
    all_output['Pack terminal voltage [V]'] = np.asarray(V_terminal)
    all_output['Cell current [A]'] = shm_i_app[:step+1, :]

    for j in range(Nvar):
        all_output[variable_names[j]] = output[j, :step+1, :]

    # Stop the Dask distributed scheduler
    client.close()
    print(client)

    return all_output
