#
# Test some experiments
#
import pybamm
import numpy as np
import unittest
import liionpack as lp


class Test1p1s(unittest.TestCase):
    def test_consistent_results_1_step(self):
        Rsmall = 1e-6
        netlist = lp.setup_circuit(
            Np=1, Ns=1, Rb=Rsmall, Rc=Rsmall, Ri=5e-2, V=4.0, I=1.0
        )
        # Heat transfer coefficients
        htc = np.ones(2) * 10
        # PyBaMM parameters
        chemistry = pybamm.parameter_sets.Chen2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        # Cycling experiment
        experiment = pybamm.Experiment(
            [("Discharge at 1 A for 100 s or until 3.3 V",)] * 1, period="10 s"
        )
        SoC = 0.5
        # Solve pack
        output = lp.solve(
            netlist=netlist,
            parameter_values=parameter_values,
            experiment=experiment,
            htc=htc,
        )

        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        parameter_values.update({"Total heat transfer coefficient [W.m-2.K-1]": 10.0})
        sim = lp.create_simulation(parameter_values, experiment, make_inputs=False)

        sol = sim.solve(initial_soc=SoC)
        a = output["Terminal voltage [V]"].flatten()
        b = sol["Terminal voltage [V]"].entries[1:]

        assert np.allclose(a, b)

    def test_consistent_results_2_step(self):
        Rsmall = 1e-6
        netlist = lp.setup_circuit(
            Np=1, Ns=1, Rb=Rsmall, Rc=Rsmall, Ri=5e-2, V=4.0, I=1.0
        )
        # Heat transfer coefficients
        htc = np.ones(2) * 10
        # PyBaMM parameters
        chemistry = pybamm.parameter_sets.Chen2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        # Cycling experiment
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1 A for 100 s or until 3.3 V",
                    "Rest for 100 s",
                )
            ]
            * 1,
            period="10 s",
        )
        SoC = 0.5
        # Solve pack
        output = lp.solve(
            netlist=netlist,
            parameter_values=parameter_values,
            experiment=experiment,
            htc=htc,
        )

        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        parameter_values.update({"Total heat transfer coefficient [W.m-2.K-1]": 10.0})
        sim = lp.create_simulation(parameter_values, experiment, make_inputs=False)

        sol = sim.solve(initial_soc=SoC)
        a = output["Terminal voltage [V]"].flatten()
        b = sol["Terminal voltage [V]"].entries[1:]
        diff = np.abs(a - b)
        assert np.all(diff < 0.05)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
