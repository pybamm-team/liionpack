import liionpack as lp
import matplotlib.pyplot as plt
import unittest


class plotsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_draw_circuit(self):
        # net = lp.setup_circuit(Np=3, Ns=1, Rb=1e-4,
        #                        Rc=1e-2, Ri=5e-2, V=3.2, I=80.0)
        # lp.draw_circuit(net)
        # plt.close('all')
        
        # cannot get this working on github actions
        pass

if __name__ == '__main__':
    unittest.main()