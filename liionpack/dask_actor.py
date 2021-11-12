#
# Dask Actor
#

import pybamm
import liionpack as lp
import numpy as np
from liionpack.solver_utils import _create_casadi_objects as cco
from liionpack.solver_utils import _serial_step as ss


class dask_actor:
    r"""
    Each actor persists on a worker process and retains the state of each
    solution of the simulation.
    """

    def __init__(self, **kwargs):
        I_init = kwargs["I_init"]
        htc_init = kwargs["htc_init"]
        Nspm = kwargs["Nspm"]
        self.output_variables = kwargs["output_variables"]
        self.parameter_values = kwargs["parameter_values"]
        initial_soc = kwargs["initial_soc"]
        sim = lp.create_simulation(self.parameter_values, make_inputs=True)
        inputs = lp.build_inputs_dict(
            [
                I_init,
            ],
            [
                htc_init,
            ],
        )
        sol_init = sim.solve(
            t_eval=[0, 1e-6],
            starting_solution=None,
            inputs=inputs[0],
            initial_soc=initial_soc,
        ).last_state
        self.sim = sim
        self._state = np.asarray([sol_init] * Nspm, dtype=object)
        self._out = np.zeros([len(self.output_variables), Nspm])
        self.Nspm = Nspm

    def step(self, dt=10, inputs=None):
        for i in range(self.Nspm):
            self._state[i] = self.sim.step(
                dt=dt,
                starting_solution=self._state[i],
                save=False,
                npts=2,
                inputs=inputs[i],
            ).last_state
            # To do - instead of processed variables could generate functions
            # To speed up evaluation
            # At the very least we need the Terminal voltage and OCP to do
            # an equivalent circuit with liionpack
            for vi, var in enumerate(self.output_variables):
                self._out[vi, i] = self._state[i][var].entries[-1]

        return self._out


class liionpack_actor:
    def __init__(self, **kwargs):
        I_init = kwargs["I_init"]
        htc_init = kwargs["htc_init"]
        dt = kwargs["dt"]
        Nspm = kwargs["Nspm"]
        initial_soc = kwargs["initial_soc"]
        self.output_variables = kwargs["output_variables"]
        self.parameter_values = kwargs["parameter_values"]
        # integrator arguments - create serial integrator for this actor
        nproc = 1
        mapped = False
        self.simulation = lp.create_simulation(self.parameter_values, make_inputs=True)
        lp.update_init_conc(self.simulation, SoC=initial_soc)
        integrator, variables_fn, t_eval = cco(
            I_init,
            htc_init,
            self.simulation,
            dt,
            Nspm,
            nproc,
            self.output_variables,
            mapped,
        )
        self.integrator = integrator
        self.variables_fn = variables_fn
        self.t_eval = t_eval
        self.simulation.build()
        self.solution = None
        self.step_solutions = [None] * Nspm

    def step(self, dt=10, inputs=None):
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
        return self.var_eval
