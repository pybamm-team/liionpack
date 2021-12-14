#
# General utility functions
#

from scipy.interpolate import interp1d


def interp_current(df):
    """
    Returns an interpolation function for current w.r.t time

    Args:
        df (pandas.DataFrame or Dict):
            Contains data for 'Time' and 'Cells Total Current' from which to
            construct an interpolant function

    Returns:
        function:
            interpolant function of total cell current with time.

    """
    t = df["Time"]
    I = df["Cells Total Current"]
    f = interp1d(t, I)
    return f


def _convert_dict_to_list_of_dict(inputs_dict):
    """
    Convert a dictionary with multiple keys (used as model inputs) into a list
    of individual dictionaries containing one element for each key

    Args:
        inputs_dict (dict):
            a dictionary with multiple keys (used as model inputs), values are
            arrays of input values for each battery.

    Returns:
        list:
            individual dictionaries containing one element for each key

    """
    keys = inputs_dict.keys()
    dicts = []
    for values in zip(*list(inputs_dict.values())):
        dicts.append(dict(zip(keys, values)))
    return dicts


def build_inputs_dict(I_batt, inputs):
    """
    Function to convert inputs and external_variable arrays to list of dicts
    As expected by the casadi solver. These are then converted back for mapped
    solving but stored individually on each returned solution.
    Can probably remove this process later

    Args:
        I_batt (np.ndarray):
            The input current for each battery.
        inputs (dict):
            A dictionary with key of each input and value an array of input
            values for each battery.

    Returns:
        list:
            each element of the list is an inputs dictionary corresponding to each
            battery.


    """
    inputs_dict = {"Current function [A]": I_batt}
    if inputs is not None:
        inputs_dict.update(inputs)
    inputs_dict = _convert_dict_to_list_of_dict(inputs_dict)
    return inputs_dict


def add_events_to_model(model):
    """
    Convert model events into variables to be evaluated in the solver step.

    Args:
        model (pybamm.lithium_ion.BaseModel):
            The PyBaMM model to solve.

    Returns:
        pybamm.lithium_ion.BaseModel:
            The PyBaMM model to solve with events added as variables.

    """
    for event in model.events:
        model.variables.update({"Event: " + event.name: event.expression})
    return model
