import liionpack as lp
import pybamm
import unittest


class simulationsTest(unittest.TestCase):

    def test_create_simulation(self):
        sim = lp.create_simulation()
        sim.solve([0, 1800])
        assert sim.__class__ == pybamm.Simulation


if __name__ == '__main__':
    unittest.main()

