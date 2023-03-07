#
# A basic example of a pack simulation with varying external temperature.
#

import liionpack as lp
import pybamm
import numpy as np
import time as ticker
from tqdm import tqdm


lp.set_logging_level("NOTICE")

# Define parameters
Np = 16
Ns = 2
Nspm = Np * Ns
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
parameter_values = pybamm.ParameterValues("Chen2020")
T0 = parameter_values["Initial temperature [K]"]

inputs = {}
input_temperature = np.ones(Nspm) * T0
inputs.update({"Input temperature [K]": input_temperature})
# Solve the pack

rm = lp.CasadiManager()
rm.solve(
    netlist=netlist,
    sim_func=lp.thermal_external,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    inputs=inputs,
    nproc=2,
    initial_soc=0.5,
    setup_only=True,
)


def external_stepper(manager, T0):
    tic = ticker.time()
    # Do stepping
    lp.logger.notice("Starting step solve")
    vlims_ok = True
    input_temperature = np.ones(Nspm) * T0
    with tqdm(total=manager.Nsteps, desc="Stepping simulation") as pbar:
        step = 0
        while step < manager.Nsteps and vlims_ok:
            input_temperature += 0.1
            inputs.update({"Input temperature [K]": input_temperature})
            vlims_ok = manager._step(step, inputs)
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
