#
# Simulate a big circuit
#

import liionpack as lp
import numpy as np
import pybamm
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.close("all")
    lp.logger.setLevel("NOTICE")
    managers = ["casadi", "ray", "dask"]
    Np = 32
    Ns = 12
    Nspm = Np * Ns
    I_app = Np * 2.0

    # Heat transfer coefficients
    htc = np.ones(Nspm) * 10

    # Cycling experiment
    experiment = pybamm.Experiment(
        [
            f"Discharge at {I_app} A for 500 seconds",
        ],
        period="10 seconds",
    )

    # PyBaMM parameters
    chemistry = pybamm.parameter_sets.Chen2020
    parameter_values = pybamm.ParameterValues(chemistry=chemistry)

    # Solve pack

    netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1e-4, Rc=1e-2, Ri=5e-2, V=3.2, I=I_app)

    output = lp.solve(
        netlist=netlist,
        parameter_values=parameter_values,
        experiment=experiment,
        output_variables=None,
        htc=htc,
        nproc=12,
        initial_soc=0.5,
        manager=managers[1],
    )

    lp.plot_output(output)
