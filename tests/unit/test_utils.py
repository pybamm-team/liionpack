import liionpack as lp
import pandas as pd
import numpy as np
import pybamm
import unittest


class utilsTest(unittest.TestCase):
    def test_interp_current(self):
        d = {"Time": [0, 10], "Cells Total Current": [2.0, 4.0]}
        df = pd.DataFrame(data=d)
        f = lp.interp_current(df)
        assert f(5) == 3.0

    def test_read_cfd_data_linear(self):
        data, xv, yv, planes = lp.read_cfd_data()
        expected = np.array([8.295482896342516, 7.878096887315587])
        T = np.ones(32) * xv[1, 1]
        Q = yv[1, 1]
        htc = lp.get_linear_htc(planes, T, Q)
        assert np.allclose(htc[:2], expected)

    def test_read_cfd_data_interpolated(self):
        data, xv, yv, funcs = lp.read_cfd_data(fit="interpolated")
        expected = np.array([8.295482896342516, 7.878096887315587])
        T = np.ones(32) * xv[1, 1]
        Q = yv[1, 1]
        htc = lp.get_interpolated_htc(funcs, T, Q)
        assert np.allclose(htc[:2], expected)

    def test_add_events_to_model(self):
        model = pybamm.lithium_ion.SPMe()
        model = lp.add_events_to_model(model)
        events_in = False
        for key in sorted(model.variables.keys()):
            if "Event:" in key:
                events_in = True
                break
        assert events_in

if __name__ == "__main__":
    unittest.main()
