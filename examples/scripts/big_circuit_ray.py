#
# Simulate a big circuit
#

import liionpack as lp
import numpy as np
import pybamm
import matplotlib.pyplot as plt
import ray
import time as ticker


print('Initializing Ray')
ray.init(dashboard_port=8080)
print('Ray initialization Complete')

plt.close("all")
lp.logger.setLevel("NOTICE")

Np = 64
Ns = 100
Nspm = Np * Ns
I_app = Np * 2.0


output_variables = [
    "X-averaged total heating [W.m-3]",
    "Volume-averaged cell temperature [K]",
    "X-averaged negative particle surface concentration [mol.m-3]",
    "X-averaged positive particle surface concentration [mol.m-3]",
]

# Heat transfer coefficients
htc = np.ones(Nspm) * 10

# Cycling experiment
experiment = pybamm.Experiment(
    [
        f"Discharge at {I_app} A for 50 seconds",
        # "Rest for 100 seconds",
        # f"Charge at {I_app} A for 500 seconds",
        # "Rest for 100 seconds",
    ],
    period="10 seconds",
)

# PyBaMM parameters
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

# Solve pack
t1 = ticker.time()
rm = lp.ray_manager.remote(Np=Np, Ns=Ns, Rb=1e-4, Rc=1e-2, Ri=5e-2, V=3.2, I=I_app)
t2 = ticker.time()
print('Ray setup time', t2 - t1)
output = ray.get(rm.solve.remote(
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=None,
    htc=htc,
    nproc=10,
))
t3 = ticker.time()
print('Ray solve time', t3 - t2)


lp.plot_output(output)

ray.shutdown()
