import liionpack as lp
import pybamm
import unittest


class simulationsTest(unittest.TestCase):
    def test_basic_simulation(self):
        sim = lp.basic_simulation()
        sim.solve([0, 1800])
        assert sim.__class__ == pybamm.Simulation

    def test_thermal_simulation(self):
        sim = lp.thermal_simulation()
        sim.solve(
            [0, 1800], inputs={"Total heat transfer coefficient [W.m-2.K-1]": 1.0}
        )
        assert sim.__class__ == pybamm.Simulation

    # def test_thermal_external(self):
    #     sim = lp.thermal_external()
    #     sim.solve([0, 1800],
    #               external_variables={"Volume-averaged cell temperature": 1.0})
    #     assert sim.__class__ == pybamm.Simulation


if __name__ == "__main__":
    unittest.main()
