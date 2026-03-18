import liionpack as lp
import pybamm
import matplotlib.pyplot as plt
from build_battery import (
    base_model,
    advanced_model,
    discret_points,
    param_base,
    param_adv,
)

# print("pybamm version: ", pybamm.__version__) # must be 24.9


def sim_base(parameter_values=None):
    global base_model
    # Add events to the model
    base_model = lp.add_events_to_model(base_model)
    # Set up simulation
    sim = pybamm.Simulation(
        model=base_model,
        parameter_values=parameter_values,
        solver=pybamm.CasadiSolver(mode="safe"),
        var_pts=discret_points,
    )
    return sim


def sim_advanced(parameter_values=None):
    global advanced_model
    # Add events to the model
    advanced_model = lp.add_events_to_model(advanced_model)
    # Set up simulation
    sim = pybamm.Simulation(
        model=advanced_model,
        parameter_values=parameter_values,
        solver=pybamm.CasadiSolver(mode="safe"),
        var_pts=discret_points,
    )
    return sim


I_mag = 160.0
OCV_init = 3.5  # used for initial guess
Ri_init = 6e-4  # used for initial guess
R_busbar = 5e-5
R_connection = 1e-4
Np = 4
Ns = 1
Nbatt = Np * Ns
netlist = lp.setup_circuit(
    Np=Np, Ns=Ns, Rb=R_busbar, Rc=R_connection, Ri=Ri_init, V=OCV_init, I=I_mag
)

oneC = param_base["Nominal cell capacity [A.h]"]
rate = 1
current = rate * oneC * Nbatt  # 1C in A
initial_soc = 0.99
final_soc = 0.55
time = 60 * (initial_soc - final_soc) / rate  # in minutes

experiment = pybamm.Experiment(
    [
        # f"Discharge at {current} A for {time} minutes",
        f"Discharge at 1000 W for {time} minutes",
        # "Rest for 15 minutes",
        # "Discharge at 5 A for 30 minutes",
        "Rest for 100 minutes",
    ],
    period=f"{time / 50} minutes",
)

output_variables = [
    # "X-averaged negative particle surface concentration",
    # "X-averaged positive particle surface concentration",
    # "X-averaged negative electrode extent of lithiation",
    "X-averaged positive electrode extent of lithiation"
]

output_base = lp.solve(
    sim_func=sim_base,
    netlist=netlist,
    parameter_values=param_base,
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=initial_soc,
)

output_adv = lp.solve(
    sim_func=sim_advanced,
    netlist=netlist,
    parameter_values=param_adv,
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=initial_soc,
)

outputs = [output_base, output_adv]
styles = ["--", "-"]
labels = ["Base model", "Adv. model"]

print("Output keys:")
print(output_adv.keys())


# Convenient function to compute SOC from extent of lithiation
def get_soc(output):
    sto_at_100 = 0.005
    sto_at_0 = 0.813
    ext_of_lith = output["X-averaged positive electrode extent of lithiation"]
    return 100 * (1 - (ext_of_lith - sto_at_100) / (sto_at_0 - sto_at_100))


fig, ax = plt.subplots()
for out, style, lab in zip(outputs, styles, labels):
    time = out["Time [s]"]
    soc = get_soc(out)
    for cell_number in range(Nbatt):
        soc_cell = soc[:, cell_number]
        ax.plot(
            time / 60,
            soc_cell,
            style,
            label=f"{lab} - Cell {cell_number + 1}",
            color=f"C{cell_number}",
        )

ax.set_xlabel("Time (min)", fontsize=14)
ax.set_ylabel("State of Charge (%)", fontsize=14)
ax.legend(frameon=False)

# lp.plot_cells(output_adv, color="light")
# lp.plot_cells(output_base, color="dark")

plt.show()
