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
        list:
            a sequence of terminal currents to apply at each timestep

    """
    protocol = []
    for i, op in enumerate(experiment.operating_conditions):
        proto = []
        t = op["time"]
        dt = op["period"]
        if t % dt != 0:
            raise ValueError("Time must be an integer multiple of the period")
        I, typ = op["electric"]
        if typ != "A":
            raise ValueError("Only constant current operations are supported")
        if I.__class__ is str:
            # drive cycle
            dc_data = op["dc_data"]
            proto.extend(dc_data[:, 1].tolist())
        elif I.__class__ is np.ndarray:
            # drive cycle old
            proto.extend(I[:, 1].tolist())
        else:
            proto.extend([I] * int(t / dt))
            if i == 0:
                # Include initial state when not a drive cycle for first op
                proto = [proto[0]] + proto
        if flatten:
            protocol.extend(proto)
        else:
            protocol.append(proto)

    return protocol
