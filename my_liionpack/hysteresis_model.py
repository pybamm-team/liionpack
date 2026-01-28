import matplotlib.pyplot as plt
import time
from my_liionpack.params_and_ocps.base_battery_param import *
import pickle



model = pybamm.lithium_ion.DFN(options = {"particle": ("quadratic profile","uniform profile"),
                                                #  "thermal": "lumped",
                                                "open-circuit potential": ("one-state hysteresis",
                                                                            "one-state hysteresis"),
                                                "particle size": ("single","distribution"),                         
                                                 },
                                    name="Sigmoid OCPs")

slow_charge = pybamm.Experiment(
        [   
            f"Charge at {param_battery["Nominal cell capacity [A.h]"]/10} A for 6000 minutes or until 3.5 V",
            "Rest for 360 minutes",
        ],
    )

slow_discharge =  pybamm.Experiment(
        [   
            f"Discharge at {param_battery["Nominal cell capacity [A.h]"]/10} A for 6000 minutes or until 2.5 V",
            "Rest for 360 minutes",
        ],
    )

experiments = [slow_charge, slow_discharge]

labels = ['adv', 'base']
inital_socs = [0.1, 0.9]
solutions = []
for exp in experiments:
    index = experiments.index(exp)

    sim = pybamm.Simulation(
            model = model,
            parameter_values = param_base,
            solver = pybamm.IDAKLUSolver(),
            var_pts = discret_points,
            experiment = exp,
        )

    current_time = time.time()
    sol = sim.solve(initial_soc=inital_socs[index])
    print("Simulation adv:", time.time() - current_time, " sec")
    solutions.append(sol)



outputs = [
    "Voltage [V]",
    "Current [A]",
    "Positive particle surface concentration",
    "Negative particle surface concentration",
    "X-averaged positive electrode hysteresis state",
]


pybamm.dynamic_plot(
    solutions,
    output_variables=outputs,
)
