#
# General utility functions
#

import numpy as np
import pathlib
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


def build_inputs_dict(I_batt, inputs, updated_inputs):
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
        updated_inputs (dict):
            A dictionary with key of each updated input and value an array
            of variable values for each battery.

    Returns:
        list:
            each element of the list is an inputs dictionary corresponding to each
            battery.


    """
    inputs_dict = {}
    current_dict = {"Current function [A]": I_batt}
    inputs_dict.update(current_dict)
    if inputs is not None:
        inputs_dict.update(inputs)
    if updated_inputs is not None:
        inputs_dict.update(updated_inputs)
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


def save_to_csv(output, path="./csv-results"):
    """
    Save simulation output to a CSV file for each output variable.

    Parameters
    ----------
    output : dict
        Simulation output dictionary.
    path : str
        Folder path for saving the CSV files. Default path is a folder named
        `csv-results` in the current directory.

    Returns
    -------
        CSV files written to the specified path. Each file represents a single
        output variable.
    """

    # Create folder path for saving files
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True)

    # Save simulation output to CSV files
    for k, v in output.items():
        filename = k.replace(" ", "_") + ".csv"
        np.savetxt(path / filename, v, delimiter=", ")


def save_to_npy(output, path="./npy-results"):
    """
    Save simulation output to NumPy `.npy` files where each file represents an
    output variable.

    Parameters
    ----------
    output : dict
        Simulation output dictionary.
    path : str
        Folder path where the `.npy` files are saved. Default path is a folder
        named `npy-results` located in the current directory.

    Returns
    -------
        NumPy `.npy` files written to the specified path. Each file represents
        a single output variable.
    """

    # Create folder path for saving files
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True)

    # Save simulation output to npy files
    for k, v in output.items():
        filename = k.replace(" ", "_") + ".npy"
        np.save(path / filename, v)


def save_to_npzcomp(output, path="."):
    """
    Save simulation output to a compressed NumPy `output.npz` file. The saved
    file is a dictionary-like object where each key represents a simulation
    output variable.

    Parameters
    ----------
    output : dict
        Simulation output dictionary.
    path : str
        Path where the `output.npz` file is saved. Default path is the current
        directory.

    Returns
    -------
        A compressed NumPy `.npz` file named `output.npz` written to the
        specified path. The file is a dictionary-like object where each key
        has the same name as the simulation output variable.
    """

    # Create a path for saving the file
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True)

    # Save simulation output to a compressed npz file
    filename = "output.npz"
    np.savez_compressed(path / filename, **output)
