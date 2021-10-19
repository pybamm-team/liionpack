"""
Dask solver functions.
"""

import casadi
import liionpack as lp
import numpy as np
import pybamm
from dask.distributed import Client
from tqdm import tqdm


def _create_casadi_objects(I_init, htc, sim, dt, Nspm, nproc, variable_names):
    """
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
    """
    inputs = {"Current": I_init,
              "Total heat transfer coefficient [W.m-2.K-1]": htc}
    solver = sim.solver
    # solve model for 1 second to initialise the circuit
    t_eval = np.linspace(0, 1, 2)

    # Initial solution - this builds the model behind the scenes
    sim.solve(t_eval, inputs=inputs)

    # step model
    # Code to create mapped integrator
    t_eval = np.linspace(0, dt, 11)
    t_eval_ndim = t_eval / sim.model.timescale.evaluate()
    inp_and_ext = inputs

    # No external variables - Temperature solved as lumped model in pybamm
    # External variables could (and should) be used if battery thermal problem
    # Includes conduction with any other circuits or neighboring batteries
    # inp_and_ext.update(external_variables)
    integrator = solver.create_integrator(sim.built_model, inputs=inp_and_ext, t_eval=t_eval_ndim)

    # Variables function for parallel evaluation
    casadi_objs = sim.built_model.export_casadi_objects(variable_names=variable_names)
    variables = casadi_objs['variables']
    t, x, z, p = casadi_objs["t"], casadi_objs["x"], casadi_objs["z"], casadi_objs["inputs"]

    variables_stacked = casadi.vertcat(*variables.values())
    variables_fn = casadi.Function("variables", [t, x, z, p], [variables_stacked])

    return integrator, variables_fn, t_eval


def _step_solve(input_dict, solution, model, integrator, variables, t_eval):
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
        htc=None, initial_soc=0.5, nproc=12, output_variables=None):
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
    integrator, variables_fn, t_eval = _create_casadi_objects(I_init, htc[0], sim, dt, Nspm, nproc, variable_names)

    # Calculate solutions for each time step
    for step in tqdm(range(Nsteps), desc='Solving Pack'):

        inputs_dict = lp.build_inputs_dict(shm_i_app[step, :], htc)

        # Calculate solution for each battery cell model
        lazy_results = client.map(
            _step_solve, inputs_dict, step_solutions,
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
