import liionpack as lp
import numpy as np
import matplotlib.pyplot as plt
import unittest


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
        V_node, I_batt = lp.solve_circuit(netlist)
        V_node_v, I_batt_v = lp.solve_circuit_vectorized(netlist)
        assert np.allclose(V_node, V_node_v)

    def test_setup_circuit_terminals(self):
        combos = [
            ["left", "right", "left-right", "right-left"],
            [[0, 0], [-1, -1], [0, -1], [-1, 0]],
        ]
        expected = [[0, 0], [7, 7], [0, 7], [7, 0]]
        for terminals in combos:
            for i, t in enumerate(terminals):
                netlist = lp.setup_circuit(Np=7, Ns=1, Rb=1e-4, terminals=t)
                I_src = netlist[netlist["desc"] == "I0"]
                assert I_src["node1_x"].item() == expected[i][0]
                assert I_src["node2_x"].item() == expected[i][1]

        netlist = lp.setup_circuit(Np=7, Ns=1, Rb=1e-4, terminals=[4, 4])
        I_src = netlist[netlist["desc"] == "I0"]
        assert I_src["node1_x"].item() == 4
        assert I_src["node2_x"].item() == 4

    def test_terminals_exception(self):
        def bad_terminals():
            _ = lp.setup_circuit(Np=7, Ns=1, Rb=1e-4, terminals="bad")

        with self.assertRaises(ValueError):
            bad_terminals()


if __name__ == "__main__":
    unittest.main()
