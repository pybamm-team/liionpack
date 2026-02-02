from main_src import *
import pybamm
import liionpack as lp

# This code needs the old pybamm version and liionpack installed

model = pybamm.lithium_ion.SPMe()

params = pybamm.ParameterValues("Prada2013")

rate = 1
initial_soc = 0.9
total_time = 0.4*abs(3600/rate)

delta_t = 10  # seconds
time_steps = int(total_time/delta_t)
# Pack configuration
num_cells_parallel = 4
# configuration = 'U' by default in my pack

r_busbar = 50*2.5e-5  # Busbar resistance between cells
r_terminal = 1e-4  # Terminal resistance

capacity = params["Nominal cell capacity [A.h]"]
total_current = rate * capacity * num_cells_parallel  
print(f"Total current for circuit solver: {total_current/num_cells_parallel} A")

experiment = pybamm.Experiment(
    [
    # f"Rest for {100*60} seconds",
    f"Discharge at {total_current} A for {total_time} seconds or until 2.2 V",
    f"Rest for {10*60} seconds",
    # f"Charge at {total_current/2} A for {2*total_time} seconds or until 4.2 V",
    ],
    period=f"{delta_t} seconds",)

sols, history = run_pack(
    model = model, 
    parameters = params,
    experiment = experiment,
    num_cells_parallel = num_cells_parallel,
    r_busbar = r_busbar,
    r_terminal = r_terminal,
    initial_soc = initial_soc,
    save_memory = True,
    solver = pybamm.CasadiSolver(mode='safe'),
    # solver = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-4),
)

print("Pack simulation complete.")

# pybamm.dynamic_plot(
#     sols, 
#     output_variables=["Voltage [V]", "Current [A]", "Positive particle surface concentration"],
# )

print("Simulation complete.")

def plot_results(history, num_cells):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    
    time = np.array(history["time"])

    # Plot Currents
    for i in range(num_cells):
        axes[0].plot(time+time[1], history["currents"][i], label=f'Cell {i+1}', color=f'C{i}')
    axes[0].set_title("Cell Currents")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Current [A]")
    axes[0].grid(True, alpha=0.5)

    # Plot Voltages
    for i in range(num_cells):
        axes[1].plot(time+time[1], history["cell_voltages"][i], label=f'Cell {i+1}', color=f'C{i}')
    axes[1].set_title("Terminal Voltages")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Voltage [V]")
    axes[1].grid(True, alpha=0.5)
    # axes[1].plot(time, history["terminal_voltage"], 'o', label='Mine Terminal V', alpha=0.7)

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

print("Now running liionpack circuit solver for verification...")
def sim_ida(parameter_values=None):
    global model
    # Add events to the model
    model = lp.add_events_to_model(model)
    # Set up simulation
    sim = pybamm.Simulation(
        model=model,
        parameter_values=parameter_values,
        # solver = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-4),
        solver = pybamm.CasadiSolver(mode='safe'),
    )
    return sim

netlist = lp.setup_circuit(
    Np=num_cells_parallel, 
    Ns=1, 
    Rb=r_busbar, 
    Rc=1e-5, 
    Ri=0.02, 
    Rt=r_terminal,
    V=3.8, 
    I=total_current,
    terminals='left'
)

out = lp.solve(
    sim_func = sim_ida,
    netlist=netlist,
    parameter_values=params,
    experiment=experiment,
    output_variables=["Voltage [V]", "Current [A]"],
    initial_soc=initial_soc,
)


t = out["Time [s]"]
volts = out["Terminal voltage [V]"]
v_term = out["Pack terminal voltage [V]"]
currents = out["Current [A]"]
R_i = out['Cell internal resistance [Ohm]']
print(out.keys())

for i in range(num_cells_parallel):
    ax[0].plot(t, currents[:,i], '--', label='Liionpack', alpha=0.7, color='C'+str(i))
    ax[1].plot(t, volts[:,i], '--', label='Liionpack', alpha=0.7,color='C'+str(i))
    ax[2].plot(t, R_i[:,i], '--', label='Liionpack', alpha=0.7, color='C'+str(i))
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
# ax[1].plot(t, v_term, 'o', label='Liionpack Terminal Voltage', alpha=0.7)

plt.tight_layout()
plt.show()