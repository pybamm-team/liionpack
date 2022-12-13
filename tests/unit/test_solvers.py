import liionpack as lp
import pybamm
import numpy as np
import unittest


class solversTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        Np = 21
        Ns = 1
        self.Nspm = Np * Ns
        R_bus = 1e-4
        R_series = 1e-2
        R_int = 5e-2
        I_app = 20.0
        ref_voltage = 3.2
        # Generate the netlist
        self.netlist = lp.setup_circuit(
            Np, Ns, Rb=R_bus, Rc=R_series, Ri=R_int, V=ref_voltage, I=I_app
        )

        # Heat transfer coefficients
        self.htc = np.ones(self.Nspm) * 10
        # Cycling experiment
        self.experiment = pybamm.Experiment(
            [
                f"Discharge at {I_app} A for 300 seconds",
            ],
            period="10 seconds",
        )
        # PyBaMM parameters
        self.parameter_values = pybamm.ParameterValues("Chen2020")
        self.managers = ["casadi", "ray"]

    def test_multiprocessing(self):
        for manager in self.managers:
            output1 = lp.solve(
                netlist=self.netlist.copy(),
                parameter_values=self.parameter_values,
                experiment=self.experiment,
                output_variables=None,
                inputs=None,
                initial_soc=0.5,
                nproc=1,
                manager=manager,
            )
            output2 = lp.solve(
                netlist=self.netlist.copy(),
                parameter_values=self.parameter_values,
                experiment=self.experiment,
                output_variables=None,
                inputs=None,
                initial_soc=0.5,
                nproc=2,
                manager=manager,
            )
            a = output1["Terminal voltage [V]"]
            b = output2["Terminal voltage [V]"]
            self.assertEqual(a.shape, (31, 21))
            self.assertTrue(np.allclose(a, b))

    def test_events(self):
        for manager in self.managers:
            output1 = lp.solve(
                netlist=self.netlist.copy(),
                parameter_values=self.parameter_values,
                experiment=self.experiment,
                output_variables=None,
                inputs=None,
                initial_soc=0.01,
                nproc=1,
                manager=manager,
            )
            output2 = lp.solve(
                netlist=self.netlist.copy(),
                parameter_values=self.parameter_values,
                experiment=self.experiment,
                output_variables=None,
                inputs=None,
                initial_soc=0.01,
                nproc=2,
                manager=manager,
            )
            a = output1["Terminal voltage [V]"]
            b = output2["Terminal voltage [V]"]
            self.assertEqual(a.shape, (8, 21))
            self.assertTrue(np.allclose(a, b))

    def test_voltage_limits(self):
        I_app = 5.0
        netlist = lp.setup_circuit(
            Np=1, Ns=1, Rb=1e-4, Rc=1e-2, Ri=3e-2, V=3.6, I=I_app
        )
        parameter_values = pybamm.ParameterValues("Chen2020")
        # Cycling experiment
        charge_exp = pybamm.Experiment(
            [
                f"Charge at {I_app} A for 10000 seconds",
            ],
            period="100 seconds",
        )
        discharge_exp = pybamm.Experiment(
            [
                f"Discharge at {I_app} A for 10000 seconds",
            ],
            period="100 seconds",
        )
        _ = lp.solve(
            netlist=netlist.copy(),
            parameter_values=parameter_values,
            experiment=charge_exp,
            output_variables=None,
            inputs=None,
            initial_soc=0.5,
            nproc=1,
            manager="casadi",
        )
        _ = lp.solve(
            netlist=netlist.copy(),
            parameter_values=parameter_values,
            experiment=discharge_exp,
            output_variables=None,
            inputs=None,
            initial_soc=0.5,
            nproc=1,
            manager="casadi",
        )
        assert True

    # def test_external_variable(self):
    #     T_non_dim = np.zeros(self.Nspm)  # Ref temperature
    #     external_variables = {"Volume-averaged cell temperature": T_non_dim}
    #     output = lp.solve(
    #         netlist=self.netlist.copy(),
    #         parameter_values=self.parameter_values,
    #         sim_func=lp.thermal_external,
    #         experiment=self.experiment,
    #         output_variables=["Volume-averaged cell temperature [K]"],
    #         inputs=None,
    #         external_variables=external_variables,
    #         initial_soc=0.5,
    #         nproc=1,
    #         manager="casadi",
    #     )
    #     assert np.all(output["Volume-averaged cell temperature [K]"] == 298.15)


if __name__ == "__main__":
    unittest.main()
