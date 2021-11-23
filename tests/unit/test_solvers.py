import liionpack as lp
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import unittest


class solversTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        Np = 21
        Ns = 1
        Nspm = Np * Ns
        R_bus = 1e-4
        R_series = 1e-2
        R_int = 5e-2
        I_app = 80.0
        ref_voltage = 3.2
        # Generate the netlist
        self.netlist = lp.setup_circuit(
            Np, Ns, Rb=R_bus, Rc=R_series, Ri=R_int, V=ref_voltage, I=I_app
        )

        # Heat transfer coefficients
        self.htc = np.ones(Nspm) * 10
        # Cycling experiment
        self.experiment = pybamm.Experiment(
            [
                f"Charge at {I_app} A for 300 seconds",
            ],
            period="10 seconds",
        )
        # PyBaMM parameters
        chemistry = pybamm.parameter_sets.Chen2020
        self.parameter_values = pybamm.ParameterValues(chemistry=chemistry)

    def test_CasadiManager(self):
        output1 = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            output_variables=None,
            htc=self.htc,
            initial_soc=0.5,
            nproc=1,
            manager="casadi",
        )
        output2 = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            output_variables=None,
            htc=self.htc,
            initial_soc=0.5,
            nproc=4,
            manager="casadi",
        )
        a = output1["Terminal voltage [V]"]
        b = output2["Terminal voltage [V]"]
        self.assertEqual(a.shape, (30, 32))
        self.assertTrue(np.allclose(a, b))

        plt.close("all")

    def test_RayManager(self):
        output1 = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            output_variables=None,
            htc=self.htc,
            initial_soc=0.5,
            nproc=1,
            manager="ray",
        )
        output2 = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            output_variables=None,
            htc=self.htc,
            initial_soc=0.5,
            nproc=4,
            manager="ray",
        )
        a = output1["Terminal voltage [V]"]
        b = output2["Terminal voltage [V]"]
        self.assertEqual(a.shape, (30, 32))
        self.assertTrue(np.allclose(a, b))

        plt.close("all")

    def test_DaskManager(self):
        output1 = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            output_variables=None,
            htc=self.htc,
            initial_soc=0.5,
            nproc=1,
            manager="dask",
        )
        output2 = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values,
            experiment=self.experiment,
            output_variables=None,
            htc=self.htc,
            initial_soc=0.5,
            nproc=4,
            manager="dask",
        )
        a = output1["Terminal voltage [V]"]
        b = output2["Terminal voltage [V]"]
        self.assertEqual(a.shape, (30, 32))
        self.assertTrue(np.allclose(a, b))

        plt.close("all")


if __name__ == "__main__":
    unittest.main()
