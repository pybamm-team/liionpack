from main_src import *
import pybamm
import liionpack as lp
from simple_plot import plot_results
from my_liionpack.params_and_ocps.base_battery_param import *
import pickle

# I think that with this new method we can save nicely charged states and discharged states
# and initialize the models from them. 
with open("my_liionpack/models/Charged_adv_model.pkl", "rb") as f:
    adv_model_charged = pickle.load(f)
with open("my_liionpack/models/Discharged_adv_model.pkl", "rb") as f:
    adv_model_discharged = pickle.load(f)

with open("my_liionpack/models/Charged_base_model.pkl", "rb") as f:
    base_model_charged = pickle.load(f)
with open("my_liionpack/models/Discharged_base_model.pkl", "rb") as f:
    base_model_discharged = pickle.load(f)

rate = 3
# initial_soc = 0.99
# total_time = 0.2*abs(3600/rate)
delta_t = 60  # seconds
num_cells_parallel = 4
r_busbar = 2.5e-5  # Busbar resistance between cells
r_terminal = 1e-6  # Terminal resistance
terminals = "left"   # choose from:
# "left", "right", "left-right", "right-left", "middle"

capacity = param_base["Nominal cell capacity [A.h]"]
total_current = rate * capacity * num_cells_parallel  

experiment = pybamm.Experiment(
    [
    # f"Charge at {capacity * num_cells_parallel/2} A until 3.5 V",
    # f"Rest for {10*60} seconds",
    # f"Discharge at {total_current} A for {total_time} seconds or until 2.6 V",
    # f"Rest for {10*60} seconds",
    # f"Charge at {total_current/2} A for {2*total_time} seconds or until 4.2 V",

    # f"Charge at {capacity * num_cells_parallel/10} A for 600 minutes or until 3.5 V",
    "Rest for 3 minutes",
    f"Discharge at {total_current} A for {60*0.5/rate} minutes",
    # "Rest for 60 minutes",
    # f"Charge at {total_current/2} A for {60*0.25/rate} minutes",
    "Rest for 6 minutes",
    ],
    period=f"{delta_t} seconds",)

outputs = [
    "Voltage [V]",
    "Current [A]",
    "X-averaged positive electrode extent of lithiation",
    "Positive particle surface concentration",
    "Negative particle surface concentration",
]

sols_base, history_base = run_pack(
    model = base_model_charged, 
    parameters = param_base,
    experiment = experiment,
    num_cells_parallel = num_cells_parallel,
    r_busbar = r_busbar,
    r_terminal = r_terminal,
    terminals = terminals,
    # save_memory = True, # this reduces memory usage but does not allow to plot all variables
    var_pts = discret_points,
    # solver = pybamm.CasadiSolver(mode='safe'),
    Ri = 0.005,
)

sols_adv, history_adv = run_pack(
    model = adv_model_charged,
    parameters = param_adv,
    experiment = experiment,
    num_cells_parallel = num_cells_parallel,
    r_busbar = r_busbar,
    r_terminal = r_terminal,
    terminals = terminals,
    # save_memory = True, # this reduces memory usage but does not allow to plot all variables
    var_pts = discret_points,
    # solver = pybamm.CasadiSolver(mode='safe'),
    Ri = 0.005,
)

print("Pack simulation complete.")

sols = []
sols.extend(sols_base)
sols.extend(sols_adv)

# pybamm.dynamic_plot(
#     sols,
#     output_variables=outputs,
# )

print("Simulation complete.")
fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True)
plot_results(history_base, num_cells_parallel, fig, axes, linestyle='--', lab = 'Base')
plot_results(history_adv, num_cells_parallel, fig, axes, linestyle='-', lab = 'Adv')

plt.tight_layout()
plt.show()