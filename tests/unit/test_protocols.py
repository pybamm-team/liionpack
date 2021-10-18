import pybamm
import liionpack as lp
import numpy as np
import unittest


class protocolsTest(unittest.TestCase):

    def test_generate_protocol(self):
        experiment = pybamm.Experiment(
            ["Charge at 50 A for 30 minutes",
            "Rest for 15 minutes",
            "Discharge at 50 A for 30 minutes",
            "Rest for 15 minutes"],
            period="10 seconds",
        )
        p = lp.generate_protocol_from_experiment(experiment)
        print(len(p))
        self.assertEqual(len(p), 540)
        self.assertEqual(np.sign(p[0]), -1)

        experiment = pybamm.Experiment(
            ["Discharge at 50 A for 30 minutes",
            "Rest for 15 minutes",
            "Charge at 50 A for 30 minutes",
            "Rest for 15 minutes"],
            period="10 seconds",
        )
        p = lp.generate_protocol_from_experiment(experiment)
        self.assertEqual(len(p), 540)
        self.assertEqual(np.sign(p[0]), 1)

if __name__ == '__main__':
    unittest.main()
