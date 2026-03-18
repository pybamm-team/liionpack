import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time
from ocps import *

# Script to build the model akin to the battery of Stock et al 2023.
# The thermal model slows down everything by 5 times.
# Maybe better to think about an easier model like: take (V - OCV)*I as heat source at each timestep and impose the temperature in the cell.

# Also, at the moment the advance model has differnt OCVs, better to change the one of prada to get the same at 0.1C

base_model = pybamm.lithium_ion.DFN(
    options={
        "particle": ("quadratic profile", "uniform profile"),
        #  "thermal": "lumped",
        "open-circuit potential": ("current sigmoid", "current sigmoid"),
    },
    name="Sigmoid OCPs",
)

advanced_model = pybamm.lithium_ion.DFN(
    options={
        "particle": ("quadratic profile", "uniform profile"),
        # "thermal": "lumped",
    },
    name="Phase field OCPs",
)

discret_points = {
    "x_n": 20,
    "x_s": 3,
    "x_p": 100,
    "r_p": 10,
    "r_n": 10,
}


def electrolyte_diffusivity_Nyman2008_arrhenius(c_e, T):
    """
    .. [1] A. Nyman, Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.
    .. [2] Ecker, Madeleine, et al. Journal of the Electrochemical Society 162.9 (2015): A1836-A1848.
    """

    D_c_e = 8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10

    E_D_c_e = 17000
    arrhenius = np.exp(E_D_c_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius


def electrolyte_conductivity_Nyman2008_arrhenius(c_e, T):
    sigma_e = (
        0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000)
    )

    E_sigma_e = 17000
    arrhenius = np.exp(E_sigma_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius


def LFP_ecd(c_e, c_s_surf, c_s_max, T):
    k_0 = 0.1
    E_r = 39570
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    c_e_ref = 1000

    return (
        (k_0 / c_s_max)
        * arrhenius
        * pybamm.sqrt(c_e / c_e_ref)
        * pybamm.sqrt(c_s_surf / c_s_max)
        * (c_s_max - c_s_surf)
    )


def graphite_ecd(c_e, c_s_surf, c_s_max, T):
    """
    Kieran O’Regan, Electrochimica Acta 425 (2022): 140700
    """

    i_ref = 2.668  # (A/m2)
    alpha = 0.792
    E_r = 4e4
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    c_e_ref = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")

    return (
        i_ref
        * arrhenius
        * (c_e / c_e_ref) ** (1 - alpha)
        * (c_s_surf / c_s_max) ** alpha
        * (1 - c_s_surf / c_s_max) ** (1 - alpha)
    )


def graphite_diffusivity_Chen2020(sto, T):
    D_ref = 3.3e-14
    E_D_s = 3.03e4
    # E_D_s not given by Chen et al (2020), so taken from Ecker et al. (2015) instead
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


param_battery = pybamm.ParameterValues("Prada2013")

# update geometry parameters to match Stock et al 2023
param_battery.update(
    {
        # Electrode properties
        "Negative electrode thickness [m]": 71e-6,
        "Negative electrode porosity": 0.20,
        "Negative electrode active material volume fraction": 0.61,
        "Negative particle radius [m]": 6.5e-6,
        "Initial concentration in negative electrode [mol.m-3]": 0.78
        * 30555,  # Based on the voltage curves in the paper
        "Separator thickness [m]": 13e-6,
        "Separator porosity": 0.508,
        "Positive electrode thickness [m]": 94e-6,
        "Positive electrode porosity": 0.32,
        "Positive electrode active material volume fraction": 0.59,
        "Positive particle radius [m]": 100e-9,
        "Initial concentration in positive electrode [mol.m-3]": 0.005 * 22806,
        # Cell properties
        "Electrode height [m]": 0.067,  # to give an area of 0.18 m2
        "Electrode width [m]": 22 * 4,
        "Nominal cell capacity [A.h]": 161.5,
        "Current function [A]": 161.5,
        # Operation limits
        "Lower voltage cut-off [V]": 2.5,
        "Upper voltage cut-off [V]": 3.7,
        "Open-circuit voltage at 0% SOC [V]": 2.5,
        "Open-circuit voltage at 100% SOC [V]": 3.7,
        # Electrolyte from Nyman 2008
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.2594,
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Nyman2008_arrhenius,
        "Electrolyte conductivity [S.m-1]"
        "": electrolyte_conductivity_Nyman2008_arrhenius,
        # For heating
        "Negative current collector thickness [m]": 2.5e-6,
        "Positive current collector thickness [m]": 6e-6,
        "Negative current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector conductivity [S.m-1]": 36914000.0,
        "Negative current collector density [kg.m-3]": 8933.0,
        "Positive current collector density [kg.m-3]": 2702.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative electrode density [kg.m-3]": 1500.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 1437.0,  # Ecker 2015
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.58,  # Ecker 2015
        "Separator density [kg.m-3]": 1017.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 1978.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.34,
        "Positive electrode density [kg.m-3]": 2400.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 1200.0,  # to be checked
        "Positive electrode thermal conductivity [W.m-1.K-1]": 1.5,  # to be checked
        "Negative electrode OCP entropic change [V.K-1]": 0.0,  # to be calculated
        "Positive electrode OCP entropic change [V.K-1]": 0.0,  # to be calculated
        "Cell cooling surface area [m2]": 0.091,
        "Cell volume [m3]": 1.411e-3,
        "Initial temperature [K]": 298.15,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,  # this could be tuned
        "Ambient temperature [K]": 298.15,  # this is the one that have to change in order to make inter-cell heating
        # "Exchange-current density for lithium metal electrode [A.m-2]": 1e3,
    },
    check_already_exists=False,
)

# Kinetic parameters
param_battery.update(
    {
        "Positive electrode conductivity [S.m-1]": 0.1,  # Ombrini 2025
        "Positive electrode exchange-current density [A.m-2]": LFP_ecd,
        "Negative particle diffusivity [m2.s-1]": graphite_diffusivity_Chen2020,  # from Chen 2020
        "Negative electrode exchange-current density [A.m-2]": graphite_ecd,
        "Negative electrode Bruggeman coefficient (electrolyte)": 2.8,  # from Dickmanns 2025
    }
)

param_base = param_battery.copy()
param_base.update(
    {
        "Negative electrode OCP [V]": graphite_ocp_avg,
        "Negative electrode lithiation OCP [V]": graphite_ocp_lithi,
        "Negative electrode delithiation OCP [V]": graphite_ocp_delithi,
        "Positive electrode OCP [V]": LFP_ocp_avg,
        "Positive electrode lithiation OCP [V]": LFP_ocp_lithi,
        "Positive electrode delithiation OCP [V]": LFP_ocp_delithi,
    },
    check_already_exists=False,
)

param_adv = param_battery.copy()
param_adv.update(
    {
        "Negative electrode OCP [V]": graphite_ocp_phase_field,
        "Positive electrode OCP [V]": LFP_ocp_phase_field,
    }
)


# if this is main script, run a test simulation
if __name__ == "__main__":
    rate = 0.5
    initial_state_of_charge = 0.001
    current = rate * param_battery["Nominal cell capacity [A.h]"]  # 1C in A

    experiment = pybamm.Experiment(
        [
            # f"Discharge at {current} A until {param_battery["Lower voltage cut-off [V]"]} V",
            "Rest for 10 minutes",
            f"Discharge at {2 * current} A for {0.5 * 10 / rate} minutes",
            f"Charge at {2 * current} A for {0.5 * 10 / rate} minutes",
            "Rest for 200 minutes",
            f"Discharge at {current} A for {30 / rate} minutes",
            "Rest for 50 minutes",
            f"Charge at {current} A for {20 / rate} minutes",
            "Rest for 50 minutes",
            f"Discharge at {current} A for {40 / rate} minutes",
            # f"Charge at {current} A for {1/rate} minutes",
            # "Rest for 60 minutes",
            # "Rest for 60 minutes",
            # f"Charge at {current} A for {30/rate} minutes",
            # "Rest for 50 minutes",
            # f"Discharge at {current} A until {param_battery['Lower voltage cut-off [V]']} V",
            # "Rest for 50 minutes",
            # f"Charge at {current} A until {param_battery["Upper voltage cut-off [V]"]} V",
        ],
    )

    # Set up solver and simulation
    sim_adv = pybamm.Simulation(
        model=advanced_model,
        parameter_values=param_adv,
        solver=pybamm.CasadiSolver(mode="safe"),
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
        solver=pybamm.CasadiSolver(mode="safe"),
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
        output_variables=output_variables,
    )

    print("Plotting voltage components of the advanced model...")
    sol_adv.plot_voltage_components(split_by_electrode=True)

    print("Plotting voltage components of the base model...")
    sol_base.plot_voltage_components(split_by_electrode=True)

    # Rate tests, to be expanded with temperature variations
    print("Running rate tests...")
    plt.figure()
    for rate in [0.1, 0.5, 1]:
        print(f"Running rate test at: {rate}C")
        current = rate * param_battery["Nominal cell capacity [A.h]"]  # 1C in A

        experiment = pybamm.Experiment(
            [
                f"Discharge at {current} A until {param_battery['Lower voltage cut-off [V]']} V",
                f"Hold at {param_battery['Lower voltage cut-off [V]']} V until C/20",
                f"Charge at {current} A until {param_battery['Upper voltage cut-off [V]']} V",
            ],
        )

        sim_adv = pybamm.Simulation(
            model=advanced_model,
            parameter_values=param_adv,
            solver=pybamm.CasadiSolver(mode="safe"),
            var_pts=discret_points,
            experiment=experiment,
        )

        current_time = time.time()
        sol = sim_adv.solve()
        print(
            f"Rate: {rate}C, Simulation time: ", time.time() - current_time, " seconds"
        )

        plt.plot(
            sol["Discharge capacity [A.h]"].entries
            / param_battery["Nominal cell capacity [A.h]"],
            sol["Voltage [V]"].entries,
            label=f"{rate}C (adv)",
        )

        sim_base = pybamm.Simulation(
            model=base_model,
            parameter_values=param_base,
            solver=pybamm.CasadiSolver(mode="safe"),
            var_pts=discret_points,
            experiment=experiment,
        )

        current_time = time.time()
        sol = sim_base.solve()
        print(
            f"Rate: {rate}C, Simulation time: ", time.time() - current_time, " seconds"
        )
        plt.plot(
            sol["Discharge capacity [A.h]"].entries
            / param_battery["Nominal cell capacity [A.h]"],
            sol["Voltage [V]"].entries,
            "--",
            label=f"{rate}C (base)",
        )

    plt.xlabel("Depth of Discharge")
    plt.ylabel("Voltage [V]")
    plt.legend()

    plt.show()
