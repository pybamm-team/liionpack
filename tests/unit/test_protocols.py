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
        p, t, s = lp.generate_protocol_from_experiment(experiment)
        self.assertEqual(np.sign(p[0][0]), -1)

        experiment = pybamm.Experiment(
            [
                "Discharge at 50 A for 30 minutes",
                "Rest for 15 minutes",
                "Charge at 50 A for 30 minutes",
                "Rest for 15 minutes",
            ],
            period="10 seconds",
        )
        p, t, s = lp.generate_protocol_from_experiment(experiment)
        self.assertEqual(len(p), 4)
        self.assertEqual(np.sign(p[0][0]), 1)

    def test_generate_protocol_from_drive_cycle(self):
        os.chdir(pybamm.__path__[0] + "/..")
        drive_cycle = pd.read_csv(
            pybamm.DataLoader().get_data("US06.csv"), comment="#", header=None
        ).to_numpy()

        experiment = pybamm.Experiment(
            [pybamm.step.current(drive_cycle)], period="1 second"
        )
        p, t, s = lp.generate_protocol_from_experiment(experiment)
        assert len(p[0]) == 601
        assert np.allclose(np.mean(p), 0.8404807891846922)

    def test_current_exception(self):
        def bad_current():
            experiment = pybamm.Experiment(
                [
                    "Charge at 1 C for 30 seconds",
                ],
                period="10 seconds",
            )
            _ = lp.generate_protocol_from_experiment(experiment)

        with self.assertRaises(ValueError):
            bad_current()

    def test_termination(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at 50 A for 30 minutes or until 3.0 V",
            ],
            period="10 seconds",
        )
        p, t, s = lp.generate_protocol_from_experiment(experiment)
        assert t[0] == 3.0


if __name__ == "__main__":
    unittest.main()
