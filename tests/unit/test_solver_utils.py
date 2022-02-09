import liionpack as lp
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import unittest


class solver_utilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        Np = 16
        Ns = 2
        self.Nspm = Np * Ns
        R_bus = 1e-4
        R_series = 1e-2
        R_int = 5e-2
        I_app = 80.0
        ref_voltage = 3.2
        # Generate the netlist
        self.netlist = lp.setup_circuit(
            Np, Ns, Rb=R_bus, Rc=R_series, Ri=R_int, V=ref_voltage, I=I_app
        )

        # Cycling experiment
        self.experiment = pybamm.Experiment(
            [
                f"Charge at {I_app} A for 300 seconds",
            ],
            period="10 seconds",
        )
        # PyBaMM parameters
        self.parameter_values = pybamm.ParameterValues("Chen2020")

    def test_mapped_step(self):
        pass

    def test_create_casadi_objects(self):
        pass

    def test_solve(self):
        output1 = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            output_variables=None,
            inputs=None,
            initial_soc=0.5,
            nproc=1,
        )
        output2 = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            output_variables=None,
            inputs=None,
            initial_soc=0.5,
            nproc=2,
        )
        a = output1["Terminal voltage [V]"]
        b = output2["Terminal voltage [V]"]
        self.assertEqual(a.shape, (31, 32))
        self.assertTrue(np.allclose(a, b))

        plt.close("all")

    def test_solve_output_variables(self):
        var = "X-averaged negative particle surface concentration [mol.m-3]"
        output_variables = [
            var,
        ]
        output = lp.solve(
            netlist=self.netlist,
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            output_variables=output_variables,
            inputs=None,
            initial_soc=0.5,
        )
        self.assertEqual(output[var].shape, (31, 32))
        plt.close("all")

    def test_sim_func(self):
        def bespoke_sim(parameter_values):
            model = pybamm.lithium_ion.SPM(
                options={
                    "thermal": "lumped",
                }
            )
            parameter_values.update(
                {
                    "Current function [A]": "[input]",
                    "Total heat transfer coefficient [W.m-2.K-1]": "[input]",
                },
            )

            # Set up solver and simulation
            solver = pybamm.CasadiSolver(mode="safe")
            sim = pybamm.Simulation(
                model=model,
                experiment=None,
                parameter_values=parameter_values,
                solver=solver,
            )
            return sim

        # Heat transfer coefficients
        inputs = {
            "Total heat transfer coefficient [W.m-2.K-1]": np.ones(self.Nspm) * 10
        }
        output = lp.solve(
            netlist=self.netlist,
            sim_func=bespoke_sim,
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            output_variables=None,
            inputs=inputs,
            initial_soc=0.5,
        )
        self.assertEqual(output["Terminal voltage [V]"].shape, (31, 32))
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
