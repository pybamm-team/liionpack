import matplotlib.pyplot as plt
import time
from my_liionpack.params_and_ocps.base_battery_param import *

# Script to build the model akin to the battery of Stock et al 2023.
# The thermal model slows down everything by 5 times.
# Maybe better to think about an easier model like: take (V - OCV)*I as heat source at each timestep and impose the temperature in the cell.

# Also, at the moment the advance model has differnt OCVs, better to change the one of prada to get the same at 0.1C

base_model = pybamm.lithium_ion.DFN(
    options={
        "particle": ("quadratic profile", "uniform profile"),
        #  "thermal": "lumped",
        "open-circuit potential": ("one-state hysteresis", "one-state hysteresis"),
        "particle size": ("single", "distribution"),
    },
    name="Sigmoid OCPs",
)

advanced_model = pybamm.lithium_ion.DFN(
    options={
        "particle": ("quadratic profile", "uniform profile"),
        # "thermal": "lumped",
        "particle size": ("single", "distribution"),
    },
    name="Phase field OCPs",
)

# if this is main script, run a test simulation
if __name__ == "__main__":
    rate = 2
    initial_state_of_charge = 0.01
    current = rate * param_battery["Nominal cell capacity [A.h]"]  # 1C in A

    experiment = pybamm.Experiment(
        [
            # "Discharge at 0.1 C for 600 minutes or until 2.8 V",
            # "Rest for 180 minutes",
            # f"Charge at {current} A for {60*0.5/rate} minutes",
            "Rest for 180 minutes",
            # f"Discharge at {current} A until 2.6 V",
            f"Charge at {param_battery['Nominal cell capacity [A.h]'] / 10} A for 600 minutes or until 3.5 V",
            "Rest for 180 minutes",
            *(
                [
                    f"Discharge at {current} A for {60 * 0.2 / rate} minutes",
                    "Rest for 180 minutes",
                ]
                * 2
            ),
        ],
    )

    # Set up solver and simulation
    sim_adv = pybamm.Simulation(
        model=advanced_model,
        parameter_values=param_adv,
        # solver=pybamm.CasadiSolver(mode="safe"),
        solver=pybamm.IDAKLUSolver(),
        var_pts=discret_points,
        experiment=experiment,
    )

    current_time = time.time()
    sol_adv = sim_adv.solve(initial_soc=initial_state_of_charge)
    print("Simulation adv:", time.time() - current_time, " sec")

    # Set up solver and simulation
    sim_base = pybamm.Simulation(
        model=base_model,
        parameter_values=param_base,
        # solver=pybamm.CasadiSolver(mode="safe"),
        solver=pybamm.IDAKLUSolver(),
        var_pts=discret_points,
        experiment=experiment,
    )

    current_time = time.time()
    sol_base = sim_base.solve(initial_soc=initial_state_of_charge)
    print("Simulation base:", time.time() - current_time, " sec")

    output_variables = [
        "Negative particle surface concentration",
        "Positive particle surface concentration",
        "Voltage [V]",
        "Discharge capacity [A.h]",
        # "X-averaged cell temperature [C]",
        "Electrolyte concentration [mol.m-3]",
    ]

    pybamm.dynamic_plot(
        [sol_adv, sol_base],
        # sol_base,
        output_variables=output_variables,
    )

    print("Plotting voltage components of the advanced model...")
    sol_adv.plot_voltage_components(split_by_electrode=True)

    print("Plotting voltage components of the base model...")
    sol_base.plot_voltage_components(split_by_electrode=True)

    # Rate tests, to be expanded with temperature variations
    print("Running rate tests...")
    plt.figure()
    for rate in [0.01, 1, 2]:
        print(f"Running rate test at: {rate}C")
        current = rate * param_battery["Nominal cell capacity [A.h]"]  # 1C in A

        experiment = pybamm.Experiment(
            [
                f"Discharge at {current} A for 6000 minutes or until {param_battery['Lower voltage cut-off [V]']} V",
                f"Hold at {param_battery['Lower voltage cut-off [V]']} V until C/20",
                f"Charge at {current} A for 6000 minutes or until {param_battery['Upper voltage cut-off [V]']} V",
            ],
        )

        sim_adv = pybamm.Simulation(
            model=advanced_model,
            parameter_values=param_adv,
            # solver=pybamm.CasadiSolver(mode="safe"),
            solver=pybamm.IDAKLUSolver(1e-8, 1e-8),
            var_pts=discret_points,
            experiment=experiment,
        )

        current_time = time.time()
        sol = sim_adv.solve()
        print(
            f"Rate: {rate}C, Simulation time: ", time.time() - current_time, " seconds"
        )

        plt.plot(
            100
            - 100
            * sol["Discharge capacity [A.h]"].entries
            / param_battery["Nominal cell capacity [A.h]"],
            sol["Voltage [V]"].entries,
            label=f"{rate}C (adv)",
            color="C" + str(int(rate * 10)),
        )

        sim_base = pybamm.Simulation(
            model=base_model,
            parameter_values=param_base,
            # solver=pybamm.CasadiSolver(mode="safe"),
            solver=pybamm.IDAKLUSolver(1e-8, 1e-8),
            var_pts=discret_points,
            experiment=experiment,
        )

        current_time = time.time()
        sol = sim_base.solve()
        print(
            f"Rate: {rate}C, Simulation time: ", time.time() - current_time, " seconds"
        )
        plt.plot(
            100
            - 100
            * sol["Discharge capacity [A.h]"].entries
            / param_battery["Nominal cell capacity [A.h]"],
            sol["Voltage [V]"].entries,
            "--",
            label=f"{rate}C (base)",
            color="C" + str(int(rate * 10)),
        )

    plt.xlabel("State of Charge (%)")
    plt.ylabel("Voltage [V]")
    plt.legend()

    plt.show()
