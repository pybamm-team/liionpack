#
# Experimental protocol
#

import numpy as np

def generate_protocol_from_experiment(experiment, flatten=True):
    """

    Args:
        experiment (pybamm.Experiment):
            The experiment to generate the protocol from.
        flatten (bool):
            Default is True: return all steps in one list otherwise return a
            list of lists for each operating command.

    Returns:
        protocol (list):
            a sequence of terminal currents to apply at each timestep

    """
    protocol = []
    for i, step in enumerate(experiment.operating_conditions_steps):
        proto = []
        t = step.duration
        dt = step.period
        typ = step.type
        if typ not in ["current"]:
            raise ValueError("Only constant current operations are supported")
        else:
            if typ == "current":
                if not step.is_drive_cycle:
                    I = step.value
                    proto.extend([I] * int(np.round(t, 5) / np.round(dt, 5)))
                    if i == 0:
                        # Include initial state when not drive cycle, first op
                        proto = [proto[0]] + proto
                else:
                    proto.extend(step.value.y.tolist())

        if flatten:
            protocol.extend(proto)
        else:
            protocol.append(proto)

    return protocol
