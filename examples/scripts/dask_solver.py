"""
Example of using the Dask solver.
"""

import liionpack as lp
import pybamm
import os


def main():
    # Parameters
    Np = 16
    Ns = 2
    cells = Np * Ns

    print('Np \t', Np)
    print('Ns \t', Ns)
    print('cells \t', cells)

    # Generate the netlist
    netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1e-4, Rc=1e-2, Ri=5e-2, V=3.2, I=80.0)

    output_variables = [
        'X-averaged negative particle surface concentration [mol.m-3]',
        'X-averaged positive particle surface concentration [mol.m-3]']


    # Cycling protocol
    experiment = pybamm.Experiment([
        "Charge at 50 A for 30 minutes",
        "Rest for 15 minutes",
        "Discharge at 50 A for 30 minutes",
        "Rest for 30 minutes"],
        period="10 seconds")

    # PyBaMM parameters
    chemistry = pybamm.parameter_sets.Chen2020
    parameter_values = pybamm.ParameterValues(chemistry=chemistry)

    # Solve pack using Dask solver
    output = lp.solve(netlist=netlist,
                      parameter_values=parameter_values,
                      experiment=experiment,
                      output_variables=output_variables,
                      nproc=os.cpu_count(),
                      manager="dask")

    lp.plot_pack(output)
    lp.plot_cells(output)
    lp.show_plots()


if __name__ == '__main__':
    main()
