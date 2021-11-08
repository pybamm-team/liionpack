import pybamm
import liionpack as lp
import numpy as np
import unittest
import os
import pandas as pd


class protocolsTest(unittest.TestCase):
    def test_generate_protocol(self):
        experiment = pybamm.Experiment(
            [
                "Charge at 50 A for 30 minutes",
                "Rest for 15 minutes",
                "Discharge at 50 A for 30 minutes",
                "Rest for 15 minutes",
            ],
            period="10 seconds",
        )
        p = lp.generate_protocol_from_experiment(experiment)
        self.assertEqual(len(p), 540)
        self.assertEqual(np.sign(p[0]), -1)

        experiment = pybamm.Experiment(
            [
                "Discharge at 50 A for 30 minutes",
                "Rest for 15 minutes",
                "Charge at 50 A for 30 minutes",
                "Rest for 15 minutes",
            ],
            period="10 seconds",
        )
        p = lp.generate_protocol_from_experiment(experiment)
        self.assertEqual(len(p), 540)
        self.assertEqual(np.sign(p[0]), 1)

    def test_generate_protocol_from_drive_cycle(self):
        os.chdir(pybamm.__path__[0] + "/..")
        drive_cycle = pd.read_csv(
        "pybamm/input/drive_cycles/US06.csv", comment="#", header=None
        ).to_numpy()
        
        experiment = pybamm.Experiment(
            operating_conditions=["Run US06 (A)"],
            period="1 minute",
            drive_cycles={"US06": drive_cycle},
        )
        p = lp.generate_protocol_from_experiment(experiment)
        assert len(p) == 601
        assert np.allclose(np.mean(p), 0.8404807891846922)

if __name__ == "__main__":
    unittest.main()
