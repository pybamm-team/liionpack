import liionpack as lp
import numpy as np


class protocolsTest():
    def setup_class(self):
        pass

    def test_generate_protocol(self):
        p = lp.generate_protocol()
        assert len(p) == 540
        p = lp.generate_protocol(chg_first=False)
        assert np.sign(p[0]) == -1

    def test_test_protocol(self):
        p = lp.test_protocol()
        assert len(p) == 90

if __name__ == '__main__':
    t = protocolsTest()
    t.setup_class()
    t.test_generate_protocol()
    t.test_test_protocol()
