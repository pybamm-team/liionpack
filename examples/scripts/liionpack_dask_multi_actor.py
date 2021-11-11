import pybamm
import liionpack as lp
from liionpack.solver_utils import _create_casadi_objects as cco
from liionpack.solver_utils import _serial_step as ss
import matplotlib.pyplot as plt
from dask.distributed import Client
from tqdm import tqdm
import numpy as np


class liionpack_actor:
    def __init__(self, **kwargs):
        chemistry = kwargs["chemistry"]
        I_init = kwargs["I_init"]
        htc_init = kwargs["htc_init"]
        dt = kwargs["dt"]
        Nspm = kwargs["Nspm"]
        variable_names = [
            "Terminal voltage [V]",
            "Measured battery open circuit voltage [V]",
            "Volume-averaged cell temperature [K]",
        ]
        nproc = 1
        mapped = False
        print(chemistry)
        self.parameter_values = pybamm.ParameterValues(chemistry=chemistry)
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


if __name__ == "__main__":
    plt.close("all")
    # Start client
    Nspm = 10000
    Nworkers = 10
    client = Client(n_workers=Nworkers)
    print(client.dashboard_link)
    chemistry = pybamm.parameter_sets.Chen2020
    # Create actor
    I_app = 5.0
    htc = 10.0
    Nt = 10
    dt = 10.0

    voltages = [[] for i in range(Nworkers)]
    temperatures = [[] for i in range(Nworkers)]
    spm_per_worker = int(Nspm / Nworkers)  # make sure no remainders
    futures = []
    inputs = []
    # Global inputs
    I_range = np.linspace(I_app - 1.0, I_app + 1.0, Nspm)
    HTC = np.ones(Nspm) * htc
    split_I_range = np.split(I_range, Nworkers)
    split_HTC = np.split(HTC, Nworkers)
    for i in range(Nworkers):
        pa = client.submit(
            liionpack_actor,
            actor=True,
            pure=False,
            chemistry=chemistry,
            I_init=I_app,
            htc_init=htc,
            dt=dt,
            Nspm=spm_per_worker,
        )
        futures.append(pa)
        inputs.append(lp.build_inputs_dict(split_I_range[i], split_HTC[i]))

    actors = [af.result() for af in futures]
    # Cycle through steps
    for n in tqdm(range(Nt), desc="Stepping dask actors"):
        for i, pa in enumerate(actors):
            pa.step(dt=dt, inputs=inputs[i])
        for i, pa in enumerate(actors):
            voltages[i].append(pa.voltage().result())

    v_amalg = np.hstack(([np.asarray(v) for v in voltages]))
    fig, ax = plt.subplots()
    _ = ax.imshow(v_amalg, cmap="seismic")
    ax.set_xlim(0, Nspm - 1)
    ax.set_ylim(0, Nt - 1)
    ax.set_aspect("auto")
    plt.show()
    plt.colorbar(_)
    plt.tight_layout()
