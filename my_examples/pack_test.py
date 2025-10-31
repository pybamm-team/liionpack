import liionpack as lp
import pybamm
import numpy as np
import matplotlib.pyplot as plt
from build_battery import simple_model, discret_points, param_base, param_adv 

# print("pybamm version: ", pybamm.__version__) # must be 24.9

def my_sim(parameter_values=None):
    global simple_model
    # Add events to the model
    simple_model = lp.add_events_to_model(simple_model)
    # Set up simulation
    sim = pybamm.Simulation(
        model=simple_model,
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
    Np=Np, 
    Ns=Ns, 
    Rb=R_busbar, 
    Rc=R_connection, 
    Ri=Ri_init, 
    V=OCV_init, 
    I=I_mag
)

oneC = param_base["Nominal cell capacity [A.h]"]
rate = 1
current = rate * oneC * Nbatt # 1C in A
initial_soc = 0.99
final_soc = 0.55
time = 60*(initial_soc - final_soc) / rate  # in minutes
print("time for charge: ", time, " minutes")

experiment = pybamm.Experiment(
    [
        f"Discharge at {current} A for {time} minutes",
        # "Rest for 15 minutes",
        # "Discharge at 5 A for 30 minutes",
        "Rest for 300 minutes",
    ],
    period= f"{time/50} minutes",
)
# parameter_values = pybamm.ParameterValues("Chen2020")


output_variables = [
    # "X-averaged negative particle surface concentration",
    # "X-averaged positive particle surface concentration",
    # "X-averaged negative electrode extent of lithiation",
    "X-averaged positive electrode extent of lithiation"
]

output_base = lp.solve(
    sim_func = my_sim,
    netlist=netlist,
    parameter_values=param_base,
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=initial_soc,
)

output_adv = lp.solve(
    sim_func = my_sim,
    netlist=netlist,
    parameter_values=param_adv,
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=initial_soc,
)

outputs = [output_base, output_adv]
styles = ["--", "-"]
labels = ["Base model", "Advanced model"]
print(output_adv.keys())

plt.figure()
for out, style, lab in zip(outputs, styles, labels):
    time = out["Time [s]"]
    ext_of_lith = out["X-averaged positive electrode extent of lithiation"]
    plt.plot(time/60, ext_of_lith, style, label=lab)

plt.xlabel("Time [min]")
plt.ylabel("X-averaged positive electrode extent of lithiation")
plt.legend()

# lp.plot_cells(output_adv, color="light")
# lp.plot_cells(output_base, color="dark")

plt.show()