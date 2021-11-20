# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:22:21 2021

@author: tom
"""

import liionpack as lp
import numpy as np
import pybamm
import matplotlib.pyplot as plt
import time as ticker
from tqdm import tqdm
from liionpack.solver_utils import _create_casadi_objects, _serial_step, _mapped_step
from dask.distributed import Client

if __name__ == "__main__":
    plt.close("all")
    pybamm.logger.setLevel("NOTICE")
    Np = 12
    Ns = 2
    Nspm = 12 * 2
    # Generate the netlist
    netlist = lp.setup_circuit(Np=12, Ns=2, Rb=1.5e-3, Rc=1e-2, Ri=5e-2, V=4.0, I=5.0)

    output_variables = None

    # Heat transfer coefficients
    htc = np.ones(Nspm) * 10

    # Cycling protocol
    experiment = pybamm.Experiment(
        [
            "Discharge at 5 A for 1 minutes",
            # "Rest for 10 minutes",
            # "CHarge at 5 A for 30 minutes",
            # "Rest for 10 minutes"
        ],
        period="10 seconds",
    )

    # PyBaMM parameters
    chemistry = pybamm.parameter_sets.Chen2020
    parameter_values = pybamm.ParameterValues(chemistry=chemistry)

    initial_soc = 0.5
    I_init = 1.0
    mapped = False
    nproc = 12

    if netlist is None or parameter_values is None or experiment is None:
        raise Exception("Please supply a netlist, paramater_values, and experiment")

    # Get netlist indices for resistors, voltage sources, current sources
    Ri_map = netlist["desc"].str.find("Ri") > -1
    V_map = netlist["desc"].str.find("V") > -1
    I_map = netlist["desc"].str.find("I") > -1

    # Nspm = np.sum(V_map)

    protocol = lp.generate_protocol_from_experiment(experiment)
    dt = experiment.period
    Nsteps = len(protocol)

    # Solve the circuit to initialise the electrochemical models
    V_node, I_batt = lp.solve_circuit(netlist)

    sim = lp.create_simulation(parameter_values, make_inputs=True)
    lp.update_init_conc(sim, SoC=initial_soc)

    v_cut_lower = parameter_values["Lower voltage cut-off [V]"]
    v_cut_higher = parameter_values["Upper voltage cut-off [V]"]

    # The simulation output variables calculated at each step for each battery
    # Must be a 0D variable i.e. battery wide volume average - or X-averaged for 1D model
    variable_names = [
        "Terminal voltage [V]",
        "Measured battery open circuit voltage [V]",
        "Local ECM resistance [Ohm]",
    ]
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
    end_time = dt * Nsteps
    step_solutions = [None] * Nspm
    V_terminal = []
    record_times = []

    integrator, variables_fn, t_eval = _create_casadi_objects(
        I_init, htc[0], sim, dt, Nspm, nproc, variable_names, mapped
    )

    # if mapped:
    #     step_fn = _mapped_step
    # else:
    #     step_fn = _serial_step
    # sim_start_time = ticker.time()
    # for step in tqdm(range(Nsteps), desc='Solving Pack'):
    #     step_solutions, var_eval = step_fn(sim.built_model, step_solutions,
    #                                        lp.build_inputs_dict(
    #                                            shm_i_app[step, :],
    #                                            htc
    #                                         ),
    #                                        integrator, variables_fn, t_eval)
    #     output[:, step, :] = var_eval

    #     time += dt
    #     # Calculate internal resistance and update netlist
    #     temp_v = output[0, step, :]
    #     temp_ocv = output[1, step, :]
    #     temp_Ri = np.abs(output[2, step, :])
    #     shm_Ri[step, :] = temp_Ri

    #     netlist.loc[V_map, ('value')] = temp_ocv
    #     netlist.loc[Ri_map, ('value')] = temp_Ri
    #     netlist.loc[I_map, ('value')] = protocol[step]

    #     # print('Stepping time', np.around(ticker.time()-tic, 2), 's')
    #     if np.any(temp_v < v_cut_lower):
    #         print('Low V limit reached')
    #         break
    #     if np.any(temp_v > v_cut_higher):
    #         print('High V limit reached')
    #         break
    #     # step += 1
    #     if time <= end_time:
    #         record_times.append(time)
    #         V_node, I_batt = lp.solve_circuit(netlist)
    #         V_terminal.append(V_node.max())
    #     if time < end_time:
    #         shm_i_app[step+1, :] = I_batt[:] * -1
    # all_output = {}
    # all_output['Time [s]'] = np.asarray(record_times)
    # all_output['Pack current [A]'] = np.asarray(protocol[:step+1])
    # all_output['Pack terminal voltage [V]'] = np.asarray(V_terminal)
    # all_output['Cell current [A]'] = shm_i_app[:step+1, :]
    # for j in range(Nvar):
    #     all_output[variable_names[j]] = output[j, :step+1, :]

    # toc = ticker.time()
    # pybamm.logger.notice('Solve circuit time '+
    #                       str(np.around(toc-sim_start_time, 3)) + 's')
    #     # return all_output

    # # Serial step
    # # srlout = solve(netlist=netl1,
    # #                parameter_values=parameter_values,
    # #                experiment=experiment,
    # #                output_variables=output_variables,
    # #                htc=htc, mapped=False)
    # lp.plot_pack(all_output)

    import casadi

    model = sim.built_model
    len_rhs = model.concatenated_rhs.size
    inputs_dict = lp.build_inputs_dict([I_init] * Nspm, htc)
    N = len(step_solutions)
    t_min = 0.0
    timer = pybamm.Timer()
    sol = []
    var_eval = []
    dx0 = []
    dz0 = []
    dp = []

    for k in range(N):
        if step_solutions[k] is None:
            # First pass
            x0 = model.y0[:len_rhs]
            z0 = model.y0[len_rhs:]
        else:
            x0 = step_solutions[k].y[:len_rhs, -1]
            z0 = step_solutions[k].y[len_rhs:, -1]
        temp = inputs_dict[k]
        inputs = casadi.vertcat(*[x for x in temp.values()] + [t_min])
        ninputs = len(temp.values())
        # Call the integrator once, with the grid
        dx0.append(np.array(x0))
        dz0.append(np.array(z0))
        dp.append(np.array(inputs))
        # dx0.append(x0)
        # dz0.append(z0)
        # dp.append(inputs)

    def integrate(integrator, x0, p, z0):
        casadi_sol = integrator(x0=x0, z0=z0, p=p)
        y_sol = np.array(casadi_sol["xf"])[:, -1]
        return y_sol

    client = Client()
    print(client)
    print(client.dashboard_link)
    intgs = [integrator] * Nspm
    for i in range(10):
        print(i)
        lazy_results = client.map(integrate, intgs, dx0, dp, dz0)
        results = client.gather(lazy_results)
        dx0 = results
    # Stop the Dask distributed scheduler
    client.close()
    print(client)

    # y_sols = []
    # for k in range(N):
    #     casadi_sol = integrator(
    #         x0=x0, z0=z0, p=inputs
    #     )
    #     y_sol = casadi_sol["xf"]
    #     y_sols.append(y_sol)
