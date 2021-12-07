#
# Mixed location of terminals and effect on current distribution
#

import liionpack as lp
import pybamm
import matplotlib.pyplot as plt
plt.close('all')

terminals = ["left", "right", "left-right", "right-left"]
lines = ["-","--","-.",":"]
plt.figure()
for i, t in enumerate(terminals):
    netlist = lp.setup_circuit(Np=100, Ns=1, Rb=1e-4, Rc=1e-2, terminals=t)
    chemistry = pybamm.parameter_sets.Chen2020
    param = pybamm.ParameterValues(chemistry=chemistry)
    experiment = pybamm.Experiment(
                [
                    "Discharge at 50 A for 1 minutes",
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
    
    
    plt.plot(range(100), output['Cell current [A]'][-1, :], lines[i], label=t)
plt.legend()
