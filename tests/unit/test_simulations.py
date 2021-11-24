import liionpack as lp
import pybamm
import unittest


class simulationsTest(unittest.TestCase):
    def test_basic_simulation(self):
        sim = lp.basic_simulation()
        sim.solve([0, 1800], inputs={'Current function [A]':1.0})
        assert sim.__class__ == pybamm.Simulation

    def test_create_simulation(self):
        sim = lp.create_simulation()
        sim.solve([0, 1800])
        assert sim.__class__ == pybamm.Simulation


if __name__ == "__main__":
    unittest.main()
