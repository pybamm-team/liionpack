#
# Experimental protocol
#

import numpy as np
import pybamm


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
    experiment_period = experiment.period
    for i, step in enumerate(experiment.steps):
        proto = []
        t = step.duration
        dt = step.period
        if step.period != experiment_period:
            raise ValueError("Step period must be equal to experiment period")
        step_dict = step.to_dict()
        typ = step_dict["type"]
        termination = step.termination
        if typ not in ["Current", "Power"]:
            raise ValueError("Only current operations are supported")
        else:
            if not isinstance(step.value, pybamm.Interpolant):
                I = step.value
                proto.extend([I] * int(np.round(t, 5) / np.round(dt, 5)))
                if i == 0:
                    # Include initial state when not drive cycle, first op
                    proto = [proto[0]] + proto
            else:
                ts = np.linspace(0, t, int(np.round(t, 5) / np.round(dt, 5)) + 1)
                I = step.value.evaluate(ts, inputs={"start time": 0})
                proto.extend(I.tolist())
            if len(termination) > 0:
                for term in termination:
                    if isinstance(
                        term, pybamm.experiment.step.step_termination.VoltageTermination
                    ):
                        terminations.append(term.value)
            else:
                terminations.append([])

        protocol.append(proto)
        step_types.append(typ)

    return protocol, terminations, step_types
