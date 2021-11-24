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
        # initial_conditions_from_experiment calls create_init_funcs when the pkl file
        # is not found doing it this way tests the try/except block in
        # initial_conditions_from_experiment
        lp.initial_conditions_from_experiment(self.param, SoC=0.0)
        assert os.path.isfile(self.fname)
        plt.close("all")

    def test_update_init_conc(self):
        sim = lp.basic_simulation(self.param)
        a = self.param["Initial concentration in negative electrode [mol.m-3]"]
        lp.update_init_conc(sim, SoC=0.0)
        param = sim.parameter_values
        b = param["Initial concentration in negative electrode [mol.m-3]"]
        assert a > b

    def test_initial_conditions_from_experiment(self):
        a_n, a_p = lp.initial_conditions_from_experiment(self.param, SoC=0.0)
        b_n, b_p = lp.initial_conditions_from_experiment(self.param, SoC=1.0)
        assert a_n < b_n
        assert a_p > b_p
        a_n, a_p = lp.initial_conditions_from_experiment(self.param, OCV=3.0)
        b_n, b_p = lp.initial_conditions_from_experiment(self.param, OCV=4.0)
        assert a_n < b_n
        assert a_p > b_p

    def test_initial_conditions_method(self):
        sim = lp.basic_simulation(self.param)

        lp.update_init_conc(sim, SoC=1.0, method="calculation")
        param = sim.parameter_values
        c_n_1 = param["Initial concentration in negative electrode [mol.m-3]"]
        c_p_1 = param["Initial concentration in positive electrode [mol.m-3]"]

        lp.update_init_conc(sim, SoC=1.0, method="experiment")
        param = sim.parameter_values
        c_n_2 = param["Initial concentration in negative electrode [mol.m-3]"]
        c_p_2 = param["Initial concentration in positive electrode [mol.m-3]"]

        c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]
        c_p_max = param["Maximum concentration in positive electrode [mol.m-3]"]

        self.assertAlmostEqual(c_n_1 / c_n_max, c_n_2 / c_n_max, 4)
        self.assertAlmostEqual(c_p_1 / c_p_max, c_p_2 / c_p_max, 4)

    def test_bad_method(self):
        with self.assertRaises(ValueError):
            sim = lp.basic_simulation(self.param)
            lp.update_init_conc(sim, SoC=1.0, method="bad method")

    def test_bad_soc(self):
        with self.assertRaises(ValueError):
            sim = lp.basic_simulation(self.param)
            lp.update_init_conc(sim, SoC=10.0)


if __name__ == "__main__":
    unittest.main()
