import matplotlib.pyplot as plt
import time
from my_liionpack.params_and_ocps.base_battery_param import *
import pickle

# this script is to check if the created models behave as expected

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
        "Rest for 30 minutes", # check if OCV is close
        *([f"Discharge at {current} A for {60*0.2/rate} minutes", # pulses 
        "Rest for 60 minutes",]*3),
        *([f"Charge at {current} A for {60*0.2/rate} minutes", # pulses 
        "Rest for 60 minutes",]*2),
        f"Discharge at {current} A until 2.5 V",
    ],
)

output_variables = [
    "Negative particle surface concentration",
    "Positive particle surface concentration",
    "Voltage [V]",
    "Discharge capacity [A.h]",
    # "X-averaged cell temperature [C]",
    "Electrolyte concentration [mol.m-3]",
    # "X-averaged positive electrode hysteresis state", # this only work in base model
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
