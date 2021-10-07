import liionpack as lp
import matplotlib.pyplot as plt
import numpy as np
import pybamm
import unittest


class plotsTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        R_bus=1e-4
        R_series=1e-2
        R_int=5e-2
        I_app=80.0
        ref_voltage = 3.2

        # Load the netlist
        netlist = lp.read_netlist("AMMBa", Ri=R_int, Rc=R_series, Rb=R_bus, Rl=R_bus, I=I_app, V=ref_voltage)
        Nspm = np.sum(netlist['desc'].str.find('V') > -1)


        # Heat transfer coefficients
        htc = np.ones(Nspm) * 10
        # Cycling protocol
        protocol = lp.test_protocol()
        # PyBaMM parameters
        chemistry = pybamm.parameter_sets.Chen2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        # Solve pack
        output = lp.solve(netlist=netlist,
                          parameter_values=parameter_values,
                          protocol=protocol,
                          output_variables=None,
                          htc=htc)
        self.output = output


    def test_draw_circuit(self):
        net = lp.setup_circuit(Np=3, Ns=1, Rb=1e-4,
                                Rc=1e-2, Ri=5e-2, V=3.2, I=80.0)
        lp.draw_circuit(net)
        plt.close('all')


    def test_cell_scatter_plot(self):

        X_pos = [0.080052414,0.057192637,0.080052401,0.057192662,0.080052171,0.057192208,0.080052285,0.057192264,
                 -0.034260006,-0.011396764,-0.034259762,-0.011396799,-0.034259656,-0.011397055,-0.034259716,-0.01139668,
                 0.034329391,0.01146636,0.034329389,0.011466487,0.034329301,0.011466305,0.034329448,0.011465906,
                 -0.079983086,-0.057122698,-0.079983176,-0.057123076,-0.079982958,-0.057122401,-0.079982995,-0.057122961]
    
        Y_pos = [-0.046199913,-0.033000108,-0.019799939,-0.0066001454,0.0066000483,0.019799888,0.033000056,0.046200369,
                 0.046200056,0.033000127,0.019800097,0.0065999294,-0.0065998979,-0.019800061,-0.032999967,-0.046200222,
                 -0.04620005,-0.032999882,-0.019800016,-0.0065999624,0.0065997543,0.019799885,0.033000077,0.046199929,
                 0.0462001,0.033000148,0.019800099,0.0066000627,-0.0065999586,-0.019800142,-0.032999927,-0.046199973]
        fig, ax = plt.subplots()
        data = self.output['Terminal voltage [V]'][-1, :]
        lp.cell_scatter_plot(ax, X_pos, Y_pos, c=data)
        plt.close('all')

    def test_plot_pack(self):
        lp.plot_pack(self.output)
        plt.close('all')

    def test_plot_cells(self):
        lp.plot_cells(self.output)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
