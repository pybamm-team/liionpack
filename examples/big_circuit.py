"""
Simulate a big circuit containing 384 battery cells.
"""

import liionpack as lp
import pybamm
import os
import pickle

lp.log_to_file("logger_info")


if __name__ == "__main__":
    Np = 32
    Ns = 12
    Nspm = Np * Ns
    I_app = 5

    # Generate the netlist
    netlist = lp.setup_circuit(Np=Np, Ns=Ns, I=I_app)

    # Cycling experiment
    experiment = pybamm.Experiment(
        [
            f"Discharge at {I_app} A for 500 seconds",
            "Rest for 100 seconds",
            f"Charge at {I_app} A for 500 seconds",
            "Rest for 100 seconds",
        ],
        period="10 seconds",
    )

    # PyBaMM parameters
    parameter_values = pybamm.ParameterValues("Chen2020")

    # Solve pack
    output = lp.solve(
        netlist=netlist,
        parameter_values=parameter_values,
        experiment=experiment,
        nproc=os.cpu_count(),
    )

    with open("output.pickle", "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
