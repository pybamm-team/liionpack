import liionpack as lp
import pandas as pd
import numpy as np
import unittest


class utilsTest(unittest.TestCase):

    def test_interp_current(self):
        d = {'Time': [0, 10], 'Cells Total Current': [2.0, 4.0]}
        df = pd.DataFrame(data=d)
        f = lp.interp_current(df)
        assert f(5) == 3.0

    def test_read_cfd_data(self):
        data, funcs, xv, yv, planes = lp.read_cfd_data()
        expected = np.array([8.295482896342516,
                             7.878096887315587])
        assert len(funcs) == 32
        assert funcs[0](16, 0.001)[0] == expected[0]
        T = np.ones(32)*xv[1, 1]
        Q = yv[1, 1]
        htc = lp.get_interpolated_htc(funcs, T, Q)
        assert htc[0] == expected[0]
        assert htc[1] == expected[1]
        assert np.allclose(htc[:2], expected)
        lin_htc = lp.get_linear_htc(planes, T, Q)
        assert np.allclose(lin_htc[:2], expected)

if __name__ == '__main__':
    unittest.main()