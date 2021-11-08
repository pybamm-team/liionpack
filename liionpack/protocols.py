#
# Experimental protocol
#
import numpy as np


def generate_protocol_from_experiment(experiment):
    """

    Parameters
    ----------
    experiment : :class:`pybamm.Experiment`
        The experiment to generate the protocol from.

    Returns
    -------
    proto : list
        a sequence of terminal currents to apply at each timestep

    """
    proto = []
    for op in experiment.operating_conditions:
        t = op["time"]
        dt = op["period"]
        if t % dt != 0:
            raise ValueError("Time must be an integer multiple of the period")
        I, typ = op["electric"]
        if typ != "A":
            raise ValueError("Only constant current operations are supported")
        if I.__class__ is np.ndarray:
            # drive cycle
            proto.extend(I[:, 1].tolist())
        else:
            proto.extend([I] * int(t / dt))
    return proto
