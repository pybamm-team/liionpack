import liionpack as lp
import pybamm


class simulationsTest():
    def setup_class(self):
        pass

    def test_create_simulation(self):
        sim = lp.create_simulation()
        sim.solve([0, 1800])
        assert sim.__class__ == pybamm.Simulation


if __name__ == '__main__':
    t = simulationsTest()
    t.setup_class()
    t.test_create_simulation()

