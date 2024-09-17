import liionpack as lp
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import unittest
from liionpack.solver_utils import _create_casadi_objects
import casadi


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
        # Setup
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")
        solver = pybamm.CasadiSolver()
        sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver)
        
        dt = 10.0
        Nspm = 4
        nproc = 2
        variable_names = ["Terminal voltage [V]"]
        mapped = True

        inputs = [{} for _ in range(Nspm)]  # Empty inputs for this test

        # Call the function
        result = _create_casadi_objects(inputs, sim, dt, Nspm, nproc, variable_names, mapped)

        # Assertions
        self.assertIn("integrator", result)
        self.assertIn("variables_fn", result)
        self.assertIn("t_eval", result)
        self.assertIn("event_names", result)
        self.assertIn("events_fn", result)
        self.assertIn("initial_solutions", result)

        # Check types and shapes
        self.assertIsInstance(result["integrator"], casadi.Function)
        self.assertIsInstance(result["variables_fn"], casadi.Function)
        self.assertIsInstance(result["t_eval"], np.ndarray)
        self.assertEqual(len(result["t_eval"]), 11)  # 11 points for 10 second interval
        self.assertIsInstance(result["event_names"], list)
        self.assertIsInstance(result["initial_solutions"], list)
        self.assertEqual(len(result["initial_solutions"]), Nspm)

        # Check the number of inputs
        n_inputs = result["variables_fn"].n_in()
        print(f"Number of inputs to variables_fn: {n_inputs}")
        self.assertEqual(n_inputs, 4, "Expected 4 inputs for the mapped function")
        
        # Check the names of the inputs
        expected_input_names = ['i0', 'i1', 'i2', 'i3']
        for i, expected_name in enumerate(expected_input_names):
            self.assertEqual(result['variables_fn'].name_in(i), expected_name, 
                             f"Expected input {i} to be named {expected_name}")

        # Check if the variables function is mapped
        self.assertTrue("map" in result["variables_fn"].name())

        # Check that the function is mapped to the correct number of batteries
        self.assertEqual(result["variables_fn"].n_in(), Nspm, 
                         "The function should be mapped to Nspm batteries")
        
        # Print out the names of the inputs for debugging
        for i in range(n_inputs):
            print(f"Input {i}: {result['variables_fn'].name_in(i)}")

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
