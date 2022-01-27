import liionpack as lp
import pandas as pd
import pybamm
import unittest
import numpy as np


class utilsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
