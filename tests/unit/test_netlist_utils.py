import liionpack as lp
import numpy as np
import matplotlib.pyplot as plt
import unittest
import os


class netlist_utilsTest(unittest.TestCase):
    def test_read_netlist(self):
        net1 = lp.read_netlist("4p1s", I=50.0)
        net2 = lp.read_netlist("4p1s.txt", I=50.0)
        net3 = lp.read_netlist("4p1s.cir", I=50.0)
        I_map = net1["desc"].str.find("I") > -1
        assert np.all(net1[I_map]["value"] == 50.0)
        assert np.all(net2[I_map]["value"] == 50.0)
        assert np.all(net3[I_map]["value"] == 50.0)

    def test_netlist_exception(self):
        def bad_filename():
            _ = lp.read_netlist("4p1s.bad", I=50.0)

        with self.assertRaises(FileNotFoundError):
            bad_filename()

    def test_setup_circuit(self):
        netlist = lp.setup_circuit(Np=1, Ns=2, Rb=1e-4, Rc=1e-2, Ri=1e-3, V=2.0, I=10.0)
        V_map = netlist["desc"].str.find("V") > -1
        assert np.all(netlist[V_map]["value"] == 2)

    def test_circuit_configuration(self):
        netlist = lp.setup_circuit(
            Np=2,
            Ns=3,
            Rb=1e-4,
            Rc=1e-2,
            Ri=1e-3,
            V=2.0,
            I=10.0,
            configuration="parallel-strings",
        )
        assert sum(netlist["desc"].str.find("Rb") > -1) == 2
        netlist = lp.setup_circuit(
            Np=2,
            Ns=3,
            Rb=1e-4,
            Rc=1e-2,
            Ri=1e-3,
            V=2.0,
            I=10.0,
            configuration="series-groups",
        )
        assert sum(netlist["desc"].str.find("Rb") > -1) == 4

        def bad_configuration():
            _ = lp.setup_circuit(Np=2, Ns=1, Rb=1e-4, configuration="bad")

        with self.assertRaises(ValueError):
            bad_configuration()

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

    def test_solve_circuit_seriesgroups(self):
        netlist = lp.setup_circuit(
            Np=1,
            Ns=2,
            Rb=1e-4,
            Rc=1e-2,
            Ri=1e-3,
            V=2.0,
            I=1.0,
            configuration="series-groups",
        )
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
        expected = [-1, 7, -1, -1]
        for terminals in combos:
            for i, t in enumerate(terminals):
                netlist = lp.setup_circuit(Np=7, Ns=1, Rb=1e-4, terminals=t)
                I_src = netlist[netlist["desc"] == "I0"]
                assert I_src["node1_x"].item() == expected[i]
                assert I_src["node2_x"].item() == expected[i]

        netlist = lp.setup_circuit(Np=7, Ns=1, Rb=1e-4, terminals=[4, 2])
        I_src = netlist[netlist["desc"] == "I0"]
        assert I_src["node1_x"].item() == -1
        assert I_src["node2_x"].item() == -1

    def test_terminals_exception(self):
        def bad_terminals():
            _ = lp.setup_circuit(Np=7, Ns=1, Rb=1e-4, terminals="bad")

        with self.assertRaises(ValueError):
            bad_terminals()

    def test_lcapy_circuit(self):
        l = lp.setup_circuit(Np=3, Ns=1, terminals="left")
        r = lp.setup_circuit(Np=3, Ns=1, terminals="right")
        lr = lp.setup_circuit(Np=3, Ns=1, terminals="left-right")
        rl = lp.setup_circuit(Np=3, Ns=1, terminals="right-left")
        m = lp.setup_circuit(Np=3, Ns=1, terminals=[1, 1])
        cct_l = lp.make_lcapy_circuit(l)
        cct_r = lp.make_lcapy_circuit(r)
        cct_lr = lp.make_lcapy_circuit(lr)
        cct_rl = lp.make_lcapy_circuit(rl)
        cct_m = lp.make_lcapy_circuit(m)
        assert cct_l.has_dc
        assert cct_r.has_dc
        assert cct_lr.has_dc
        assert cct_rl.has_dc
        assert cct_m.has_dc

    def test_write_netlist(self):
        net = lp.setup_circuit(Np=1, Ns=2, Rb=1e-4, Rc=1e-2, Ri=1e-3, V=2.0, I=10.0)
        cwd = os.getcwd()
        temp = os.path.join(cwd, "temp.txt")
        lp.write_netlist(net, temp)
        assert os.path.isfile(temp)
        os.remove(temp)


if __name__ == "__main__":
    unittest.main()
