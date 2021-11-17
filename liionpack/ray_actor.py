#
# Ray actors
#

import liionpack as lp
from liionpack.solver_utils import _create_casadi_objects as cco
from liionpack.solver_utils import _serial_step as ss
import matplotlib.pyplot as plt
import ray
from tqdm import tqdm
import numpy as np
import pybamm
import liionpack as lp
import time as ticker


@ray.remote(num_cpus=1)
class ray_actor:
    def __init__(self):
        pass
    
    def setup(
        self,
        Nspm,
        parameter_values,
        dt,
        I_init,
        htc_init,
        variable_names,
        index,
        # manager,
        **kwargs,
    ):
        # Create an actor
        nproc = 1
        mapped = False
        self.parameter_values = parameter_values
        self.simulation = lp.create_simulation(self.parameter_values, make_inputs=True)

        integrator, variables_fn, t_eval = cco(
            I_init, htc_init, self.simulation, dt, Nspm, nproc, variable_names, mapped
        )
        self.integrator = integrator
        self.variables_fn = variables_fn
        self.t_eval = t_eval
        self.simulation.build()
        self.solution = None
        self.step_solutions = [None] * Nspm
        self.index = index
        # self.manager = manager
        return True

    def step(self, inputs):
        # Get inputs from manager arrays
        # i_app = self.manager.actor_i_app(self.index)
        # htc = self.manager.actor_htc(self.index)
        # inputs = lp.build_inputs_dict(i_app, htc)

        step_solutions, var_eval = ss(
            self.simulation.built_model,
            self.step_solutions,
            inputs,
            self.integrator,
            self.variables_fn,
            self.t_eval,
        )
        self.step_solutions = step_solutions
        self.var_eval = np.asarray(var_eval)
        # return self.var_eval
        return True
    
    def output(self):
        return self.var_eval

    def voltage(
        self,
    ):
        return self.var_eval[0, :]

    def ocv(
        self,
    ):
        return self.var_eval[1, :]

    def temperature(self):
        return self.var_eval[2, :]


# @ray.remote(num_cpus=1)
class ray_manager:
    def __init__(self, Np, Ns, Rb, Rc, Ri, V, I):
        self.netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=Rb, Rc=Rc, Ri=Ri, V=V, I=I)

    def solve(self, nproc, parameter_values, experiment, htc, output_variables):
        netlist = self.netlist
        # Get netlist indices for resistors, voltage sources, current sources
        Ri_map = netlist["desc"].str.find("Ri") > -1
        V_map = netlist["desc"].str.find("V") > -1
        I_map = netlist["desc"].str.find("I") > -1
        Terminal_Node = np.array(netlist[I_map].node1)
        Nspm = np.sum(V_map)
        self.spm_per_worker = int(Nspm / nproc)  # make sure no remainders
        # Generate the protocol from the supplied experiment
        protocol = lp.generate_protocol_from_experiment(experiment)
        dt = experiment.period
        Nsteps = len(protocol)

        # Solve the circuit to initialise the electrochemical models
        V_node, I_batt = lp.solve_circuit_vectorized(netlist)

        # The simulation output variables calculated at each step for each battery
        # Must be a 0D variable i.e. battery wide volume average - or X-averaged for 1D model
        variable_names = [
            "Terminal voltage [V]",
            "Measured battery open circuit voltage [V]",
        ]
        if output_variables is not None:
            for out in output_variables:
                if out not in variable_names:
                    variable_names.append(out)
            # variable_names = variable_names + output_variables
        Nvar = len(variable_names)

        # Storage variables for simulation data
        self.shm_i_app = np.zeros([Nsteps, Nspm], dtype=float)
        shm_Ri = np.zeros([Nsteps, Nspm], dtype=float)
        self.output = np.zeros([Nvar, Nsteps, Nspm], dtype=float)

        # Initialize currents in battery models
        self.shm_i_app[0, :] = I_batt * -1

        # Step forward in time
        time = 0
        self.timestep = 0
        end_time = dt * Nsteps
        V_terminal = []
        record_times = []

        v_cut_lower = parameter_values["Lower voltage cut-off [V]"]
        v_cut_higher = parameter_values["Upper voltage cut-off [V]"]

        

        # Ray setup an actor for each worker
        self.actors = []

        self.htc = np.split(htc, nproc)
        self.split_index = np.split(np.arange(Nspm), nproc)
        tic = ticker.time()
        for i in range(nproc):
            self.actors.append(lp.ray_actor.remote())
        setup_futures = []
        for a in self.actors:
            # Create actor on each worker containing a simulation
            setup_futures.append(
                a.setup.remote(
                Nspm=self.spm_per_worker,
                parameter_values=parameter_values,
                dt=dt,
                I_init=self.shm_i_app[0, 0],
                htc_init=htc[0],
                variable_names=variable_names,
                index=i,
                # manager=self,
                )
            )
        setup_done = [ray.get(f) for f in setup_futures]
        toc = ticker.time()
        print('Actors setup', np.all(setup_done), 'in time', toc-tic)
            # actors.append(pa)
        sim_start_time = ticker.time()
        print("Starting step solve")
        for step in range(Nsteps):
            self.step_actors()
            self.get_actor_output(step)
            # for i, fs in enumerate(future_steps):
            #     slc = slice(i * self.spm_per_worker, (i + 1) * self.spm_per_worker)
            #     out = ray.get(fs)
            #     self.output[:, step, slc] = out

            time += dt

            # Calculate internal resistance and update netlist
            temp_v = self.output[0, step, :]
            temp_ocv = self.output[1, step, :]
            # This could be used instead of Equivalent ECM resistance which has
            # been changing definition
            temp_Ri = (temp_ocv - temp_v) / self.shm_i_app[step, :]
            # Make Ri more stable
            current_cutoff = np.abs(self.shm_i_app[step, :]) < 1e-6
            temp_Ri[current_cutoff] = 1e-12
            shm_Ri[step, :] = temp_Ri

            netlist.loc[V_map, ("value")] = temp_ocv
            netlist.loc[Ri_map, ("value")] = temp_Ri
            netlist.loc[I_map, ("value")] = protocol[step]

            # Stop if voltage limits are reached
            if np.any(temp_v < v_cut_lower):
                print("Low voltage limit reached")
                break
            if np.any(temp_v > v_cut_higher):
                print("High voltage limit reached")
                break

            if time <= end_time:
                record_times.append(time)
                V_node, I_batt = lp.solve_circuit_vectorized(netlist)
                V_terminal.append(V_node[Terminal_Node][0])
            if time < end_time:
                self.shm_i_app[step + 1, :] = I_batt[:] * -1

            self.timestep += 1
        print("Step solve finished")
        for actor in self.actors:
            ray.kill(actor)
        print("Killed actors")
        # Collect outputs
        self.all_output = {}
        self.all_output["Time [s]"] = np.asarray(record_times)
        self.all_output["Pack current [A]"] = np.asarray(protocol[: step + 1])
        self.all_output["Pack terminal voltage [V]"] = np.asarray(V_terminal)
        self.all_output["Cell current [A]"] = self.shm_i_app[: step + 1, :]
        for j in range(Nvar):
            self.all_output[variable_names[j]] = self.output[j, : step + 1, :]

        toc = ticker.time()

        lp.logger.notice(
            "Total stepping time " + str(np.around(toc - sim_start_time, 3)) + "s"
        )
        lp.logger.notice(
            "Time per step " + str(np.around((toc - sim_start_time)/Nsteps, 3)) + "s"
        )
        return self.all_output

    def actor_i_app(self, index):
        actor_indices = self.split_index[index]
        return self.shm_i_app[self.timestep, actor_indices]

    def actor_htc(self, index):
        return self.htc[index]

    def build_inputs(self):
        inputs = []
        for i in range(len(self.actors)):
            I_app = self.actor_i_app(i)
            htc = self.actor_htc(i)
            inputs.append(lp.build_inputs_dict(I_app, htc))
        return inputs

    def get_actor_output(self, step):
        t1 = ticker.time()
        futures = []
        for actor in self.actors:
            futures.append(actor.output.remote())
        for i, f in enumerate(futures):
            slc = slice(i * self.spm_per_worker, (i + 1) * self.spm_per_worker)
            out = ray.get(f)
            self.output[:, step, slc] = out
        t2 = ticker.time()
        # print('Getting output', t2 - t1)

    def step_actors(self):
        t1 = ticker.time()
        future_steps = []
        inputs = self.build_inputs()
        for i, pa in enumerate(self.actors):
            future_steps.append(pa.step.remote(inputs[i]))
        done = [ray.get(fs) for fs in future_steps]
        t2 = ticker.time()
        # print('Stepping models', t2-t1)