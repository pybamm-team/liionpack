import liionpack as lp
import pandas as pd
import numpy as np


class utilsTest():
    def setup_class(self):
        pass

    def test_interp_current(self):
        d = {'Time': [0, 10], 'Cells Total Current': [2.0, 4.0]}
        df = pd.DataFrame(data=d)
        f = lp.interp_current(df)
        assert f(5) == 3.0

    def test_read_cfd_data(self):
        f = lp.read_cfd_data()
        assert len(f) == 32
        assert f[0](16, 0.001)[0] == 8.295482896342516
        htc = lp.get_interpolated_htc(f, np.ones(32)*16, 0.001)
        assert htc[0] == 8.295482896342516
        assert htc[1] == 7.878096887315587

if __name__ == '__main__':
    t = utilsTest()
    t.setup_class()
    t.test_interp_current()
    t.test_read_cfd_data()