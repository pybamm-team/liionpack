import liionpack as lp
import numpy as np
import pandas as pd
import pathlib
import pybamm
import unittest


class utilsTest(unittest.TestCase):
    def setUp(self):
        currents = [4, 5.2, 8, 10]
        volts = np.array([[3.5, 3.6], [3.54, 3.58]])
        output = {"currents": currents, "volts": volts}

        self.currents = currents
        self.volts = volts
        self.output = output

    def test_interp_current(self):
        d = {"Time": [0, 10], "Cells Total Current": [2.0, 4.0]}
        df = pd.DataFrame(data=d)
        f = lp.interp_current(df)
        assert f(5) == 3.0

    def test_add_events_to_model(self):
        model = pybamm.lithium_ion.SPMe()
        model = lp.add_events_to_model(model)
        events_in = False
        for key in sorted(model.variables.keys()):
            if "Event:" in key:
                events_in = True
                break
        assert events_in

    def test_build_inputs_dict(self):
        I_batt = np.array([1.0, 2.0])
        inputs = {"Electrode height [m]": [3.0, 4.0]}
        external_variables = {"Volume averaged cell temperature": [5.0, 6.0]}
        in_dict = lp.build_inputs_dict(I_batt, inputs, external_variables)
        assert len(in_dict) == 2

    def test_save_to_csv(self):
        lp.save_to_csv(self.output, path=".")

        with open("currents.csv", "r") as f:
            current = float(f.readline())
        self.assertEqual(self.currents[0], current)

        with open("volts.csv", "r") as f:
            volts = list(map(float, f.readline().split(", ")))
        self.assertEqual(self.volts[0][0], volts[0])

    def test_save_to_npy(self):
        lp.save_to_npy(self.output, path=".")

        currents = np.load("currents.npy")
        self.assertEqual(self.currents[0], currents[0])

        volts = np.load("volts.npy")
        self.assertEqual(self.volts[0, 0], volts[0, 0])

    def test_save_to_npz(self):
        lp.save_to_npzcomp(self.output, path=".")

        output = np.load("output.npz")
        currents = output["currents"]
        self.assertEqual(self.currents[0], currents[0])

        output = np.load("output.npz")
        volts = output["volts"]
        self.assertEqual(self.volts[0, 0], volts[0, 0])

    def tearDown(self):
        path = pathlib.Path("currents.csv")
        path.unlink(missing_ok=True)

        path = pathlib.Path("volts.csv")
        path.unlink(missing_ok=True)

        path = pathlib.Path("currents.npy")
        path.unlink(missing_ok=True)

        path = pathlib.Path("volts.npy")
        path.unlink(missing_ok=True)

        path = pathlib.Path("output.npz")
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
