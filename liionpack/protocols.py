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
    for i, step in enumerate(experiment.operating_conditions_steps):
        proto = []
        t = step.duration
        dt = step.period
        typ = step.type
        termination = step.termination
        if typ not in ["current"]:
            raise ValueError("Only current operations are supported")
        else:
            if not step.is_drive_cycle:
                I = step.value
                proto.extend([I] * int(np.round(t, 5) / np.round(dt, 5)))
                if i == 0:
                    # Include initial state when not drive cycle, first op
                    proto = [proto[0]] + proto
            else:
                proto.extend(step.value.y.tolist())
            if len(termination) > 0:
                for term in termination:
                    if isinstance(term, pybamm.experiment.step.step_termination.VoltageTermination):
                        terminations.append(term.value)
            else:
                terminations.append([])

        protocol.append(proto)

    return protocol, terminations
