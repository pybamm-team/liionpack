import liionpack as lp
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import unittest


class solver_utilsTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        Np=16
        Ns=2
        Nspm = Np * Ns
        R_bus=1e-4
        R_series=1e-2
        R_int=5e-2
        I_app=80.0
        ref_voltage = 3.2
        # Generate the netlist
        self.netlist = lp.setup_circuit(Np, Ns, Rb=R_bus, Rc=R_series,
                                        Ri=R_int, V=ref_voltage, I=I_app)

        # Heat transfer coefficients
        self.htc = np.ones(Nspm) * 10
        # Cycling experiment
        self.experiment = lp.test_experiment()
        # PyBaMM parameters
        chemistry = pybamm.parameter_sets.Chen2020
        self.parameter_values = pybamm.ParameterValues(chemistry=chemistry)

    def test_mapped_step(self):
        pass

    def test_create_casadi_objects(self):
        pass

    def test_solve(self):
        output = lp.solve(netlist=self.netlist,
                          parameter_values=self.parameter_values,
                          experiment=self.experiment,
                          output_variables=None,
                          htc=self.htc)
        print(output['Terminal voltage [V]'].shape)
        assert output['Terminal voltage [V]'].shape == (90, 32)
        plt.close('all')
    
    def test_solve_output_variables(self):
        output_variables = [  
            'X-averaged total heating [W.m-3]',
            'Volume-averaged cell temperature [K]',
            'X-averaged negative particle surface concentration [mol.m-3]',
            'X-averaged positive particle surface concentration [mol.m-3]',
            ]
        output = lp.solve(netlist=self.netlist,
                          parameter_values=self.parameter_values,
                          experiment=self.experiment,
                          output_variables=output_variables,
                          htc=self.htc)
        assert output['X-averaged total heating [W.m-3]'].shape == (90, 32)
        plt.close('all')

if __name__ == '__main__':
    unittest.main()
