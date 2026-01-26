import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time

def generate_protocol_from_experiment(experiment):
    """
    Args:
        experiment (pybamm.Experiment):
            The experiment to generate the protocol from.

    Returns:
        protocol (list):
            a sequence of terminal currents to apply at each timestep
        terminations (list):
            a sequence voltage terminations for each step

    """
    protocol = []
    terminations = []
    step_types = []
    for i, step in enumerate(experiment.steps):
        proto = []
        t = step.duration
        dt = step.period
        termination = step.termination
        step_type = type(step).__name__.lower()
        if step_type not in ["current", "power"]:
            raise ValueError("Only current and power operations are supported")
        else:
            if not isinstance(step.value, pybamm.Interpolant):
                I = step.value
                proto.extend([I] * int(np.round(t, 5) / np.round(dt, 5)))
                if i == 0:
                    # Include initial state when not drive cycle, first op
                    proto = [proto[0]] + proto
            else:
                proto.extend(step.value.y.tolist())
            if len(termination) > 0:
                for term in termination:
                    if isinstance(
                        term, pybamm.experiment.step.step_termination.VoltageTermination
                    ):
                        terminations.append(term.value)
            else:
                terminations.append([])

        protocol.append(proto)
        step_types.append(step_type)

    return protocol, terminations, step_types, dt

def setup_sims_and_params(model, parameters, solver, var_pts, num_cells_parallel):
    if var_pts is None:
        var_pts = model.default_var_pts

    params = parameters.copy()
    params.update({"Current function [A]": "[input]"})
    v_cut_lower = params["Lower voltage cut-off [V]"]
    v_cut_higher = params["Upper voltage cut-off [V]"]
    # change cut off cell_voltages to avoid the pybamm solver stopping too early
    params.update({"Lower voltage cut-off [V]": v_cut_lower - 0.5,
                "Upper voltage cut-off [V]": v_cut_higher + 0.5})
    # Simulation settings
    sims = [pybamm.Simulation(model, parameter_values = params,
                            solver = solver.copy(),
                            var_pts = var_pts,
                            ) for _ in range(num_cells_parallel)]
    return sims, v_cut_lower, v_cut_higher, params

def create_hystory_dict(num_cells_parallel):
    history = {
        "time": [],
        "currents": [[] for _ in range(num_cells_parallel)],
        "cell_voltages": [[] for _ in range(num_cells_parallel)],
        "resistances": [[] for _ in range(num_cells_parallel)],
        "temperatures": [[] for _ in range(num_cells_parallel)],
        "terminal_voltage": [],
    }
    return history