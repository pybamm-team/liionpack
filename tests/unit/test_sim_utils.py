import liionpack as lp
import pybamm
import os
import matplotlib.pyplot as plt
import unittest


class sim_utilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        chemistry = pybamm.parameter_sets.Chen2020
        self.param = pybamm.ParameterValues(chemistry=chemistry)
        self.fname = os.path.join(lp.INIT_FUNCS, "init_funcs.pickle")

    def test_create_init_funcs(self):
        if os.path.isfile(self.fname):
            os.remove(self.fname)
        lp.create_init_funcs(self.param)
        assert os.path.isfile(self.fname)
        plt.close("all")

    def test_update_init_conc(self):
        sim = lp.create_simulation(self.param)
        a = self.param["Initial concentration in negative electrode [mol.m-3]"]
        lp.update_init_conc(sim, SoC=0.0)
        param = sim.parameter_values
        b = param["Initial concentration in negative electrode [mol.m-3]"]
        assert a > b

    def test_initial_conditions(self):
        a_n, a_p = lp.initial_conditions(SoC=0.0)
        b_n, b_p = lp.initial_conditions(SoC=1.0)
        assert a_n < b_n
        assert a_p > b_p
        a_n, a_p = lp.initial_conditions(OCV=3.0)
        b_n, b_p = lp.initial_conditions(OCV=4.0)
        assert a_n < b_n
        assert a_p > b_p


if __name__ == "__main__":
    unittest.main()
