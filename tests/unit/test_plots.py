import liionpack as lp
import matplotlib.pyplot as plt
import pybamm
import unittest
import numpy as np


class plotsTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        R_bus = 1e-4
        R_series = 1e-2
        R_int = 5e-2
        I_app = 1.0
        ref_voltage = 3.2

        # Load the netlist
        self.netlist = lp.read_netlist(
            "4p1s", Ri=R_int, Rc=R_series, Rb=R_bus, Rt=R_bus, I=I_app, V=ref_voltage
        )

        # Cycling experiment
        experiment = pybamm.Experiment(
            [
                f"Charge at {I_app} A for 300 seconds",
                "Rest for 100 seconds",
                f"Discharge at {I_app} A for 300 seconds",
                "Rest for 100 seconds",
            ],
            period="10 seconds",
        )
        # PyBaMM parameters
        parameter_values = pybamm.ParameterValues("Chen2020")
        # Solve pack
        output = lp.solve(
            netlist=self.netlist,
            parameter_values=parameter_values,
            experiment=experiment,
            output_variables=None,
            inputs=None,
            initial_soc=0.5,
        )
        self.output = output
        self.sim = pybamm.Simulation(
            model=pybamm.lithium_ion.SPM(),
            parameter_values=parameter_values,
            experiment=experiment,
        )

    def test_draw_circuit(self):
        net = lp.setup_circuit(
            Np=3, Ns=1, Rb=1e-4, Rc=1e-2, Ri=5e-2, V=3.2, I=80.0, terminals=[0, 1]
        )
        lp.draw_circuit(net)
        plt.close("all")

    def test_plot_pack(self):
        lp.plot_pack(self.output)
        plt.close("all")

    def test_plot_cells(self):
        lp.plot_cells(self.output)
        plt.close("all")

    def test_plot_output(self):
        lp.plot_output(self.output)
        plt.close("all")
        lp.plot_output(self.output, color="white")
        plt.close("all")

    def test_compare_plots(self):
        solution = self.sim.solve(initial_soc=0.5)
        lp.compare_solution_output(solution, self.output)
        lp.compare_solution_output(self.output, solution)
        plt.close("all")

    def test_plot_cell_data_image(self):
        Np = 5
        Ns = 2
        Nspm = Np * Ns
        # Generate the netlist
        netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1e-4, Ri=3e-2)
        # Define additional output variables
        output_variables = [
            "Volume-averaged cell temperature [K]",
        ]
        # Define a cycling experiment using PyBaMM
        experiment = pybamm.Experiment(
            [
                "Discharge at 5 A for 5 minutes",
            ],
            period="10 seconds",
        )
        # Define the PyBaMM parameters
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values.update(
            {"Total heat transfer coefficient [W.m-2.K-1]": "[input]"}
        )
        htc = np.random.random(Nspm) * 10.0
        inputs = {"Total heat transfer coefficient [W.m-2.K-1]": htc}
        # Solve the pack
        output = lp.solve(
            netlist=netlist,
            sim_func=lp.thermal_simulation,
            parameter_values=parameter_values,
            experiment=experiment,
            output_variables=output_variables,
            inputs=inputs,
            initial_soc=0.5,
        )
        data = output["Volume-averaged cell temperature [K]"][-1, :]
        lp.plot_cell_data_image(netlist, data)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
