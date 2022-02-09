import liionpack as lp
import pybamm
import unittest


class sim_utilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.param = pybamm.ParameterValues("Chen2020")

    def test_update_init_conc(self):
        a = self.param["Initial concentration in negative electrode [mol.m-3]"]
        lp.update_init_conc(self.param, SoC=0.0, update=False)
        b = self.param["Initial concentration in negative electrode [mol.m-3]"]
        lp.update_init_conc(self.param, SoC=0.0, update=True)
        c = self.param["Initial concentration in negative electrode [mol.m-3]"]
        assert a == b
        assert a > c

    def test_bad_soc(self):
        with self.assertRaises(ValueError):
            lp.update_init_conc(self.param, SoC=10.0)


if __name__ == "__main__":
    unittest.main()
