#
# Experimental protocol
#


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
        typ = op["type"]
        if typ not in ["current"]:
            raise ValueError("Only constant current operations are supported")
        else:
            if typ == "current":
                if "Current input [A]" in op.keys():
                    I = op["Current input [A]"]
                    proto.extend([I] * int(t / dt))
                    if i == 0:
                        # Include initial state when not drive cycle, first op
                        proto = [proto[0]] + proto
                elif "dc_data" in op.keys():
                    dc_data = op["dc_data"]
                    proto.extend(dc_data[:, 1].tolist())

        if flatten:
            protocol.extend(proto)
        else:
            protocol.append(proto)

    return protocol
