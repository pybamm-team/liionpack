from utils import *
import pybamm
import liionpack as lp
# 1. Setup Model and Parameters
model = pybamm.lithium_ion.SPM()
model_2 = pybamm.lithium_ion.SPM()

params = pybamm.ParameterValues("Chen2020")

rate = 10
total_time = 0.25*abs(3600/rate)
delta_t = 1  # seconds
time_steps = int(total_time/delta_t)
# Pack configuration
num_cells_parallel = 4
# configuration = 'U'

r_busbar = 20*2.5e-5  # Busbar resistance between cells

print("Now running liionpack circuit solver for verification...")
def sim_ida(parameter_values=None):
    global model_2
    # Add events to the model
    model_2 = lp.add_events_to_model(model_2)
    # Set up simulation
    sim = pybamm.Simulation(
        model=model_2,
        parameter_values=parameter_values,
        # solver = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-4),
        solver = pybamm.CasadiSolver(mode='safe')
    )
    return sim

capacity = params["Nominal cell capacity [A.h]"]
total_current = rate * capacity * num_cells_parallel  
print(f"Total current for circuit solver: {total_current/num_cells_parallel} A")

netlist = lp.setup_circuit(
    Np=num_cells_parallel, 
    Ns=1, 
    Rb=r_busbar, 
    Rc=1e-5, 
    Ri=0.2, 
    Rt=1e-8,
    V=3.8, 
    I=total_current,
    terminals='left'
)
experiment = pybamm.Experiment(
    [f"Discharge at {total_current} A for {total_time/60} minutes",],
    period=f"{delta_t} seconds",)

try:
    out = lp.solve(
        sim_func = sim_ida,
        netlist=netlist,
        parameter_values=params,
        experiment=experiment,
        output_variables=["Voltage [V]", "Current [A]"],
        initial_soc=1,
    )
except Exception as e:
    print("An error occurred during the circuit solver simulation:")
    print(e)
    raise e

t = out["Time [s]"]
volts = out["Terminal voltage [V]"]
currents = out["Current [A]"]
R_i = out['Cell internal resistance [Ohm]']

print("Circuit solver simulation complete.")
print("Now running liionpack pack solver...")

sols, history = run_pack(
    model=model,
    params=params,
    rate=rate,
    total_time=total_time,
    time_steps=time_steps,
    num_cells_parallel=num_cells_parallel,
    r_busbar=r_busbar,
    initial_soc=1,
    save_memory=True,
    solver = pybamm.CasadiSolver(mode='safe')
)

print("Pack simulation complete.")

# pybamm.dynamic_plot(
#     sols, 
#     output_variables=["Voltage [V]", "Current [A]", "Positive particle surface concentration"],
# )

print("Simulation complete.")

def plot_results(history, num_cells):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    time = history["time"]

    # Plot Currents
    for i in range(num_cells):
        axes[0].plot(time, history["currents"][i], label=f'Cell {i+1}', color=f'C{i}')
    axes[0].set_title("Cell Currents")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Current [A]")
    axes[0].grid(True, alpha=0.5)

    # Plot Voltages
    for i in range(num_cells):
        axes[1].plot(time, history["voltages"][i], label=f'Cell {i+1}', color=f'C{i}')
    axes[1].set_title("Terminal Voltages")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Voltage [V]")
    axes[1].grid(True, alpha=0.5)

    # Plot Resistances
    for i in range(num_cells):
        axes[2].plot(time, history["resistances"][i], label=f'Cell {i+1}', color=f'C{i}')
    axes[2].set_title("Internal Resistance (Linearized)")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Resistance [Ohm]")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.5)

    return fig, axes



fig, ax = plot_results(history, num_cells_parallel)
for i in range(num_cells_parallel):
    ax[0].plot(t, currents[:,i], 'k--', label='Circuit Solver', alpha=0.7, color='C'+str(i))
    ax[1].plot(t, volts[:,i], 'k--', label='Circuit Solver', alpha=0.7,color='C'+str(i))
    ax[2].plot(t, R_i[:,i], 'k--', label='Circuit Solver', alpha=0.7, color='C'+str(i))
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

plt.tight_layout()
plt.show()