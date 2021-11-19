#
# Test solvers give the same output
#

import pybamm
import numpy as np
import unittest
import liionpack as lp


class TestSolvers(unittest.TestCase):
    def test_consistent_results_1_step(self):
        Rsmall = 1e-6
        netlist = lp.setup_circuit(
            Np=1, Ns=1, Rb=Rsmall, Rc=Rsmall, Ri=5e-2, V=4.0, I=1.0
        )
        # Heat transfer coefficients
        htc = np.ones(1) * 10
        # PyBaMM parameters
        chemistry = pybamm.parameter_sets.Chen2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        # Cycling experiment
        experiment = pybamm.Experiment(
            [("Discharge at 1 A for 100 s or until 3.3 V",)] * 1, period="10 s"
        )
        # Solve pack with casadi
        a = lp.solve(
            netlist=netlist,
            parameter_values=parameter_values,
            experiment=experiment,
            htc=htc,
            nproc=1,
            manager="casadi"
        )
        # Solve pack with dask
        b = lp.solve(
            netlist=netlist,
            parameter_values=parameter_values,
            experiment=experiment,
            htc=htc,
            nproc=1,
            manager="dask"
        )
        # Solve pack with ray
        c = lp.solve(
            netlist=netlist,
            parameter_values=parameter_values,
            experiment=experiment,
            htc=htc,
            nproc=1,
            manager="ray"
        )

        v_a = a["Terminal voltage [V]"]
        v_b = b["Terminal voltage [V]"]
        v_c = c["Terminal voltage [V]"]

        assert np.allclose(v_a, v_b)
        assert np.allclose(v_b, v_c)


if __name__ == "__main__":
    unittest.main()
