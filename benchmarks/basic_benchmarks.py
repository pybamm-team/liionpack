# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import liionpack as lp
import pybamm


class BasicBenchmark:
    def setup(self):
        self.sim = lp.basic_simulation()

    def time_solve_model(self):
        BasicBenchmark.sim.solve([0, 1800])


class SmallPack:
    def setup(self):
        self.netlist = lp.setup_circuit(Np=2, Ns=1, Rb=1e-4, Rc=1e-2)
        chemistry = pybamm.parameter_sets.Chen2020
        self.parameter_values = pybamm.ParameterValues(chemistry=chemistry)

    def time_discharge_1cpu(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at 2 A for 5 minutes",
            ],
            period="10 seconds",
        )
        _ = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values.copy(),
            experiment=experiment,
            initial_soc=0.5,
            nproc=1,
        )

    def time_discharge_2cpu(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at 2 A for 5 minutes",
            ],
            period="10 seconds",
        )
        _ = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values.copy(),
            experiment=experiment,
            initial_soc=0.5,
            nproc=2,
        )


class MediumPack:
    def setup(self):
        self.netlist = lp.setup_circuit(Np=32, Ns=10, Rb=1e-4, Rc=1e-2)
        chemistry = pybamm.parameter_sets.Chen2020
        self.parameter_values = pybamm.ParameterValues(chemistry=chemistry)

    def time_discharge_1cpu(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at 32 A for 5 minutes",
            ],
            period="10 seconds",
        )
        _ = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values.copy(),
            experiment=experiment,
            initial_soc=0.5,
            nproc=1,
        )

    def time_discharge_2cpu(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at 32 A for 5 minutes",
            ],
            period="10 seconds",
        )
        _ = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values.copy(),
            experiment=experiment,
            initial_soc=0.5,
            nproc=2,
        )


class LargePack:
    def setup(self):
        self.netlist = lp.setup_circuit(Np=64, Ns=64, Rb=1e-4, Rc=1e-2)
        chemistry = pybamm.parameter_sets.Chen2020
        self.parameter_values = pybamm.ParameterValues(chemistry=chemistry)

    def time_discharge_1cpu(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at 64 A for 5 minutes",
            ],
            period="10 seconds",
        )
        _ = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values.copy(),
            experiment=experiment,
            initial_soc=0.5,
            nproc=1,
        )

    def time_discharge_2cpu(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at 64 A for 5 minutes",
            ],
            period="10 seconds",
        )
        _ = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values.copy(),
            experiment=experiment,
            initial_soc=0.5,
            nproc=2,
        )

    def time_long_cycle_2cpu(self):
        I_app = 64
        experiment = pybamm.Experiment(
            [
                f"Charge at {I_app} A for 20 minutes",
                "Rest for 15 minutes",
                f"Discharge at {I_app} A for 20 minutes",
                "Rest for 30 minutes",
            ]
            * 10,
            period="10 seconds",
        )
        _ = lp.solve(
            netlist=self.netlist.copy(),
            parameter_values=self.parameter_values.copy(),
            experiment=experiment,
            initial_soc=0.5,
            nproc=2,
        )
