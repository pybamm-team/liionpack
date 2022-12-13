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

    def test_input_logger_warning(self):
        with self.assertRaises(ValueError):
            param = pybamm.ParameterValues("Chen2020")
            neg_conc = 0.0
            param.update(
                {
                    "Electrode height [m]": "[input]",
                    "Initial concentration in negative electrode [mol.m-3]": neg_conc,
                },
                check_already_exists=False,
            )
            lp.update_init_conc(param, SoC=0.5, update=True)
            # a = param["Initial concentration in negative electrode [mol.m-3]"]
            # assert a == neg_conc


if __name__ == "__main__":
    unittest.main()
