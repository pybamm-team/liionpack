"""
Example of using the Dask solver. Set `nproc` to number of physical CPU
cores that are available on your computer.
"""

import liionpack as lp
import pybamm
import os


def main():
    # Parameters
    Np = 16
    Ns = 2

    # Generate the netlist
    netlist = lp.setup_circuit(Np=Np, Ns=Ns)

    # Cycling protocol
    experiment = pybamm.Experiment([
        'Charge at 20 A for 30 minutes',
        'Rest for 15 minutes',
        'Discharge at 20 A for 30 minutes',
        'Rest for 30 minutes'],
        period='10 seconds')

    # PyBaMM parameters
    parameter_values = pybamm.ParameterValues("Chen2020")

    # Solve pack using Dask solver
    output = lp.solve(
        netlist=netlist,
        parameter_values=parameter_values,
        experiment=experiment,
        initial_soc=0.5,
        nproc=os.cpu_count(),
        manager='dask')

    # Plot the pack and individual cell results
    lp.plot_pack(output)
    lp.plot_cells(output)
    lp.show_plots()


if __name__ == '__main__':
    main()
