"""
A basic example of a pack simulation consisting of two sets of 16 parallel
cells connected in series for a total of 32 cells.
"""

import liionpack as lp
import pybamm
import numpy as np
import time as ticker
from tqdm import tqdm


def T_non_dim(parameter_values, T):
    param = pybamm.LithiumIonParameters()
    Delta_T = parameter_values.evaluate(param.Delta_T)
    T_ref = parameter_values.evaluate(param.T_ref)
    return (T - T_ref) / Delta_T


lp.set_logging_level("NOTICE")

# Define parameters
Np = 16
Ns = 2
Iapp = 20

# Generate the netlist
netlist = lp.setup_circuit(Np=Np, Ns=Ns)

# Define additional output variables
output_variables = ["Volume-averaged cell temperature [K]"]

# Define a cycling experiment using PyBaMM
experiment = pybamm.Experiment(
    [
        f"Charge at {Iapp} A for 30 minutes",
        "Rest for 15 minutes",
        f"Discharge at {Iapp} A for 30 minutes",
        "Rest for 30 minutes",
    ],
    period="10 seconds",
)

# Define the PyBaMM parameters
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)
T0 = parameter_values["Initial temperature [K]"]
T0_non_dim = np.ones(Np * Ns) * T_non_dim(parameter_values, T0)
external_variables = {"Volume-averaged cell temperature": T0_non_dim}

# Solve the pack

rm = lp.casadi_manager()
rm.solve(
    netlist=netlist,
    sim_func=lp.thermal_external,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    inputs=None,
    external_variables=external_variables,
    nproc=2,
    initial_soc=0.5,
    setup_only=True
)


def external_stepper(manager, T0):
    tic = ticker.time()
    # Do stepping
    lp.logger.notice("Starting step solve")
    vlims_ok = True
    with tqdm(total=manager.Nsteps, desc="Stepping simulation") as pbar:
        step = 0
        while step < manager.Nsteps and vlims_ok:
            T0 += 0.1
            T_nd = np.ones(Np * Ns) * T_non_dim(manager.parameter_values, T0)
            external_variables = {"Volume-averaged cell temperature": T_nd}
            vlims_ok = manager._step(step, external_variables)
            step += 1
            pbar.update(1)
    manager.step = step
    toc = ticker.time()
    lp.logger.notice("Step solve finished")
    lp.logger.notice("Total stepping time " + str(np.around(toc - tic, 3)) + "s")
    lp.logger.notice(
        "Time per step " + str(np.around((toc - tic) / manager.Nsteps, 3)) + "s"
    )


external_stepper(rm, T0)
output = rm.step_output()
# Plot the pack and individual cell results
lp.plot_pack(output)
lp.plot_cells(output)
lp.show_plots()
