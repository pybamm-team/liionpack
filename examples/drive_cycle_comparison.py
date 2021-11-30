import pybamm
import os
import pandas as pd
import liionpack as lp
import matplotlib.pyplot as plt
import numpy as np


plt.close("all")

os.chdir(pybamm.__path__[0] + "/..")
netlist = lp.setup_circuit(Np=1, Ns=1, Rb=1.0e-6, Rc=1e-6, Ri=3e-2, V=3.75, I=1.0)

chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

# import drive cycle from file
drive_cycle = pd.read_csv(
    "pybamm/input/drive_cycles/US06.csv", comment="#", header=None
).to_numpy()

experiment = pybamm.Experiment(
    operating_conditions=["Run US06 (A)"],
    period="1 second",
    drive_cycles={"US06": drive_cycle},
)

# PyBaMM parameters
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

# Solve pack
output = lp.solve(
    netlist=netlist,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=None,
    initial_soc=0.5,
)

sim = pybamm.Simulation(
    model=pybamm.lithium_ion.SPM(),
    experiment=experiment,
    parameter_values=parameter_values,
)
sol = sim.solve(initial_soc=0.5)

t_pybamm = sol["Time [s]"].entries
t_liionpack = output["Time [s]"]
v_pybamm = sol["Terminal voltage [V]"].entries
v_liionpack = output["Terminal voltage [V]"]
i_pybamm = sol["Current [A]"].entries
i_liionpack = output["Cell current [A]"]
r_pybamm = np.abs(sol["Local ECM resistance [Ohm]"].entries)
r_liionpack = output["Cell internal resistance [Ohm]"]

# Liionpack lags 1 step behind
t_liionpack -= 1


sol_diff = ((v_liionpack[1:].flatten() - v_pybamm[:-1]) / v_pybamm[:-1]) * 100

with plt.rc_context(lp.lp_context):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    ax0.plot(t_pybamm[:-1], sol_diff)
    ax0.set_ylabel("Voltage difference [%]")
    ax1.plot(t_pybamm, v_pybamm, label="PyBaMM")
    ax1.plot(t_liionpack, v_liionpack, "--", label="Liionpack")
    ax1.set_ylabel("Terminal voltage [V]")
    ax2.plot(t_pybamm, i_pybamm, label="PyBaMM")
    ax2.plot(t_liionpack, i_liionpack, "--", label="Liionpack")
    ax2.set_ylabel("Current [A]")
    ax3.plot(t_pybamm, r_pybamm, label="PyBaMM")
    ax3.plot(t_liionpack, r_liionpack, "--", label="Liionpack")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Internal resistance [Ohm]")
    plt.legend()
