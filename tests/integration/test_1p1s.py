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
        # PyBaMM parameters
        parameter_values = pybamm.ParameterValues("Chen2020")
        # Cycling experiment
        experiment = pybamm.Experiment(
            [("Discharge at 1 A for 100 s or until 3.3 V",)] * 1, period="10 s"
        )
        # Solve pack
        output = lp.solve(
            netlist=netlist, parameter_values=parameter_values, experiment=experiment
        )

        parameter_values = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(
            model=pybamm.lithium_ion.SPM(),
            parameter_values=parameter_values,
            experiment=experiment,
        )

        sol = sim.solve()
        a = output["Terminal voltage [V]"].flatten()
        b = sol["Terminal voltage [V]"].entries

        assert np.allclose(a, b)

    def test_consistent_results_2_step(self):
        Rsmall = 1e-6
        netlist = lp.setup_circuit(
            Np=1, Ns=1, Rb=Rsmall, Rc=Rsmall, Ri=5e-2, V=4.0, I=1.0
        )
        # PyBaMM parameters
        parameter_values = pybamm.ParameterValues("Chen2020")
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
            initial_soc=SoC,
        )

        parameter_values = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(
            model=pybamm.lithium_ion.SPM(),
            parameter_values=parameter_values,
            experiment=experiment,
        )

        sol = sim.solve(initial_soc=SoC)
        a = output["Terminal voltage [V]"].flatten()
        b = sol["Terminal voltage [V]"].entries
        diff = np.abs(a[:20] - b[:20])
        assert np.all(diff < 0.05)


if __name__ == "__main__":
    unittest.main()
