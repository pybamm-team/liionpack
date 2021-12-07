import liionpack as lp
import numpy as np
import matplotlib.pyplot as plt
import unittest
import time as time


class netlist_utilsTest(unittest.TestCase):
    def test_read_netlist(self):
        netlist = lp.read_netlist("AMMBa", I=50.0)
        I_map = netlist["desc"].str.find("I") > -1
        assert np.all(netlist[I_map]["value"] == 50.0)

    def test_setup_circuit(self):
        netlist = lp.setup_circuit(Np=1, Ns=2, Rb=1e-4, Rc=1e-2, Ri=1e-3, V=2.0, I=10.0)
        V_map = netlist["desc"].str.find("V") > -1
        assert np.all(netlist[V_map]["value"] == 2)

    def test_setup_circuit_plot(self):
        netlist = lp.setup_circuit(
            Np=1, Ns=2, Rb=1e-4, Rc=1e-2, Ri=1e-3, V=2.0, I=10.0, plot=True
        )
        V_map = netlist["desc"].str.find("V") > -1
        assert np.all(netlist[V_map]["value"] == 2)
        plt.close("all")

    def test_solve_circuit(self):
        netlist = lp.setup_circuit(Np=1, Ns=2, Rb=1e-4, Rc=1e-2, Ri=1e-3, V=2.0, I=1.0)
        V_node, I_batt = lp.solve_circuit(netlist)
        assert np.all(I_batt) == 1.0

    def test_solve_circuit_vectorized(self):
        netlist = lp.setup_circuit(
            Np=1, Ns=100, Rb=1e-4, Rc=1e-2, Ri=1e-3, V=2.0, I=1.0
        )
        t1 = time.time()
        V_node, I_batt = lp.solve_circuit(netlist)
        t2 = time.time()
        V_node_v, I_batt_v = lp.solve_circuit_vectorized(netlist)
        t3 = time.time()
        vector_time = t3 - t2
        serial_time = t2 - t1
        assert np.allclose(V_node, V_node_v)


if __name__ == "__main__":
    unittest.main()
