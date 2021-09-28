import liionpack as lp
import pybamm
import numpy as np
import matplotlib.pyplot as plt


class solver_utilsTest():
    def setup_class(self):
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
        # Cycling protocol
        self.protocol = lp.test_protocol()
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
                          protocol=self.protocol,
                          output_variables=None,
                          htc=self.htc)
        assert output.shape == (3, 90, 32)
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
                          protocol=self.protocol,
                          output_variables=output_variables,
                          htc=self.htc)
        assert output.shape == (7, 90, 32)
        plt.close('all')

if __name__ == '__main__':
    t = solver_utilsTest()
    t.setup_class()
    t.test_mapped_step()
    t.test_create_casadi_objects()
    t.test_solve()
    t.test_solve_output_variables()
