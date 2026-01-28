import matplotlib.pyplot as plt
import time
from my_liionpack.params_and_ocps.base_battery_param import *
import pickle

# import the pre made model pickle from the models folder
with open("my_liionpack/models/Charged_adv_model.pkl", "rb") as f:
    model_adv_charged = pickle.load(f)

with open("my_liionpack/models/Charged_base_model.pkl", "rb") as f:
    model_base_charged = pickle.load(f)

models = [model_base_charged, model_adv_charged]
parameters = [param_base , param_adv]
# Now we can use this model to run a simulation starting from the pre-charged state

rate = 2
current = rate * param_battery["Nominal cell capacity [A.h]"] # 1C in A

experiment = pybamm.Experiment(
    [      
        "Rest for 180 minutes",
        *([f"Discharge at {current} A for {60*0.2/rate} minutes",
        "Rest for 30 minutes"]*4)
        # "Discharge at 0.01C for 6000 minutes or until 2.5 V",
    ],
)

output_variables = [
    "Negative particle surface concentration",
    "Positive particle surface concentration",
    "Voltage [V]",
    "Discharge capacity [A.h]",
    # "X-averaged cell temperature [C]",
    "Electrolyte concentration [mol.m-3]",
    # "X-averaged positive electrode hysteresis state",
]

solutions = []
for model, param in zip(models, parameters):
    # Set up solver and simulation
    sim = pybamm.Simulation(
        model=model,
        parameter_values=param,
        solver = pybamm.IDAKLUSolver(),
        var_pts=discret_points,
        experiment=experiment,
    )

    current_time = time.time()
    sol = sim.solve()
    print("Simulation adv:", time.time() - current_time, " sec")
    solutions.append(sol)

pybamm.dynamic_plot(
    solutions,
    output_variables=output_variables,
)
