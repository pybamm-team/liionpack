from main_src import *
import pybamm
import liionpack as lp
from simple_plot import plot_results
from run_cell import base_model, advanced_model, discret_points, param_base, param_adv 

rate = 3
initial_soc = 0.9
total_time = 0.2*abs(3600/rate)
delta_t = 10  # seconds
# Pack configuration
num_cells_parallel = 4

r_busbar = 2.5e-5  # Busbar resistance between cells
r_terminal = 1e-6  # Terminal resistance

capacity = param_base["Nominal cell capacity [A.h]"]
total_current = rate * capacity * num_cells_parallel  

experiment = pybamm.Experiment(
    [
    # f"Charge at {capacity * num_cells_parallel/2} A until 3.5 V",
    # f"Rest for {10*60} seconds",
    # f"Discharge at {total_current} A for {total_time} seconds or until 2.6 V",
    # f"Rest for {10*60} seconds",
    # f"Charge at {total_current/2} A for {2*total_time} seconds or until 4.2 V",

    f"Charge at {capacity * num_cells_parallel/10} A for 600 minutes or until 3.5 V",
    "Rest for 180 minutes",
    *([f"Discharge at {total_current} A for {60*0.2/rate} minutes",
    "Rest for 180 minutes"]*2)
    ],
    period=f"{delta_t} seconds",)

sols_base, history_base = run_pack(
    model = base_model, 
    parameters = param_base,
    experiment = experiment,
    num_cells_parallel = num_cells_parallel,
    r_busbar = r_busbar,
    r_terminal = r_terminal,
    initial_soc = initial_soc,
    save_memory = True,
    var_pts = discret_points
)

sols_adv, history_adv = run_pack(
    model = advanced_model, 
    parameters = param_adv,
    experiment = experiment,
    num_cells_parallel = num_cells_parallel,
    r_busbar = r_busbar,
    r_terminal = r_terminal,
    initial_soc = initial_soc,
    save_memory = True,
    var_pts = discret_points
)

print("Pack simulation complete.")

# pybamm.dynamic_plot(
#     sols_base,
#     output_variables=["Voltage [V]", "Current [A]", 
#                       "X-averaged positive electrode extent of lithiation"],
# )
# pybamm.dynamic_plot(
#     sols_adv,
#     output_variables=["Voltage [V]", "Current [A]", 
#                       "X-averaged positive electrode extent of lithiation"],
# )

print("Simulation complete.")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
plot_results(history_base, num_cells_parallel, fig, axes, linestyle='--', label = 'Base')
plot_results(history_adv, num_cells_parallel, fig, axes, linestyle='-', label = 'Adv')

plt.tight_layout()
plt.show()