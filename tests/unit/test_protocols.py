import liionpack as lp
import numpy as np
import unittest


class protocolsTest(unittest.TestCase):

    def test_generate_protocol(self):
        p = lp.generate_protocol()
        assert len(p) == 540
        p = lp.generate_protocol(chg_first=False)
        assert np.sign(p[0]) == -1

    def test_test_protocol(self):
        p = lp.test_protocol()
        assert len(p) == 90

if __name__ == '__main__':
    unittest.main()
