"""
Compare a drive-cycle simulation between PyBaMM and Liionpack.
"""

import pybamm
import os
import pandas as pd
import liionpack as lp
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    lp.set_logging_level("NOTICE")

    os.chdir(pybamm.__path__[0] + "/..")
    netlist = lp.setup_circuit(Np=1, Ns=1, Rb=1.0e-6, Rc=1e-6, Ri=3e-2)

    parameter_values = pybamm.ParameterValues("Chen2020")

    # import drive cycle from file
    drive_cycle = pd.read_csv(
        "pybamm/input/drive_cycles/US06.csv", comment="#", header=None
    ).to_numpy()

    timestep = 1
    drive_cycle[:, 0] *= timestep

    experiment = pybamm.Experiment(
        operating_conditions=["Run US06 (A)"],
        period=f"{timestep} seconds",
        drive_cycles={"US06": drive_cycle},
    )

    output_variables = [
        "X-averaged negative particle surface concentration [mol.m-3]",
        "X-averaged positive particle surface concentration [mol.m-3]",
    ]

    # PyBaMM parameters
    parameter_values = pybamm.ParameterValues("Chen2020")

    init_SoC = 0.5

    # Solve pack
    output = lp.solve(
        netlist=netlist,
        parameter_values=parameter_values,
        experiment=experiment,
        output_variables=output_variables,
        initial_soc=init_SoC,
        manager="casadi",
        nproc=8,
    )

    parameter_values = pybamm.ParameterValues("Chen2020")

    sim = pybamm.Simulation(
        model=pybamm.lithium_ion.SPM(),
        experiment=experiment,
        parameter_values=parameter_values,
    )

    sol = sim.solve(initial_soc=init_SoC)

    t_pybamm = sol["Time [s]"].entries
    t_liionpack = output["Time [s]"]
    v_pybamm = sol["Terminal voltage [V]"].entries
    v_liionpack = output["Terminal voltage [V]"]
    i_pybamm = sol["Current [A]"].entries
    i_liionpack = output["Cell current [A]"]
    r_pybamm = np.abs(sol["Local ECM resistance [Ohm]"].entries)
    r_liionpack = output["Cell internal resistance [Ohm]"]
    nconc_pybamm = sol[
        "X-averaged negative particle surface concentration [mol.m-3]"
    ].entries
    nconc_liionpack = output[
        "X-averaged negative particle surface concentration [mol.m-3]"
    ]
    pconc_pybamm = sol[
        "X-averaged positive particle surface concentration [mol.m-3]"
    ].entries
    pconc_liionpack = output[
        "X-averaged positive particle surface concentration [mol.m-3]"
    ]

    # The internal resistance in Liionpack is based on the previous time-step
    # so the solution lags behind. More advanced time stepping is required to
    # address this problem.
    t_liionpack -= timestep
    sol_diff = ((v_liionpack.flatten()[1:] - v_pybamm[:-1]) / v_pybamm[:-1]) * 100

    with plt.rc_context(lp.lp_context()):
        fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(
            3, 2, figsize=(12, 10), sharex=True
        )
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
        ax3.set_ylabel("Internal resistance [Ohm]")
        ax4.plot(t_pybamm, nconc_pybamm, label="PyBaMM")
        ax4.plot(t_liionpack, nconc_liionpack, "--", label="Liionpack")
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Neg particle conc. [mol.m-3]")
        ax5.plot(t_pybamm, pconc_pybamm, label="PyBaMM")
        ax5.plot(t_liionpack, pconc_liionpack, "--", label="Liionpack")
        ax5.set_xlabel("Time [s]")
        ax5.set_ylabel("Pos particle conc. [mol.m-3]")
        handles, labels = ax5.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

    plt.show()
