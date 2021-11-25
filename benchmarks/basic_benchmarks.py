# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import liionpack as lp


class BasicBenchmark:
    def setup(self):
        self.sim = lp.basic_simulation()

    def time_solve_model(self):
        BasicBenchmark.sim.solve([0, 1800])
