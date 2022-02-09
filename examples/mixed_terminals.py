#
# Mixed location of terminals and effect on current distribution
#

import liionpack as lp
import pybamm
import matplotlib.pyplot as plt

plt.close("all")

Np = 7
Ns = 1
combos = [
    ["left", "right", "left-right", "right-left"],
    [[0, 0], [-1, -1], [0, -1], [-1, 0], [3, 3]],
]
for terminals in combos:
    lines = ["-", "--", "-.", ":", ".-"]
    plt.figure()
    for i, t in enumerate(terminals):
        netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1e-4, Rc=1e-2, terminals=t)
        param = pybamm.ParameterValues("Chen2020")
        experiment = pybamm.Experiment(
            [
                f"Discharge at {Np} A for 1 minutes",
            ],
            period="10 seconds",
        )
        # Solve pack
        output = lp.solve(
            netlist=netlist,
            parameter_values=param,
            experiment=experiment,
            output_variables=None,
            initial_soc=0.5,
        )

        plt.plot(range(Np), output["Cell current [A]"][-1, :], lines[i], label=t)

    plt.legend()

lp.show_plots()
