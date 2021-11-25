# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:33:13 2021

@author: Tom
"""
import os
import pathlib

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from skspatial.objects import Plane, Points

import liionpack as lp

ROOT_DIR = str(pathlib.Path(lp.__path__[0]).parent)
MODULE_DIR = os.path.dirname(os.path.abspath(lp.__file__))
CIRCUIT_DIR = os.path.join(MODULE_DIR, "circuits")
DATA_DIR = os.path.join(MODULE_DIR, "data")
INIT_FUNCS = os.path.join(MODULE_DIR, "init_funcs")


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


def _z_from_plane(X, Y, plane):
    """
    Given X and Y and a plane provide Z
    X - temperature
    Y - flow rate
    Z - heat transfer coefficient

    Args:
        X (np.ndarray):
            x-coordinate.
        Y (np.ndarray):
            z-coordinate.
        plane (skspatial.object.Plane):
            plane returned from read_cfd_data.

    Returns:
        z (np.ndarray):
            z-coordinate.

    """
    a, b, c = plane.normal
    d = plane.point.dot(plane.normal)
    z = (d - a * X - b * Y) / c
    return z


def _fit_plane(xv, yv, dbatt):
    """
    Private method to fit plane to CFD data

    Args:
        xv (np.ndarray):
            temperature meshgrid points.
        yv (np.ndarray):
            flow_rate meshgrid points.
        dbatt (np.ndarray):
            cfd data for heat transfer coefficient.

    Returns:
        plane (skspatial.object.Plane):
            htc varies linearly with temperature and flow rate so relationship
            describes a plane

    """
    nx, ny = xv.shape
    pts = []
    for i in range(nx):
        for j in range(ny):
            pts.append([xv[i, j], yv[i, j], dbatt[i, j]])

    points = Points(pts)
    plane = Plane.best_fit(points, tol=1e-6)
    return plane


def read_cfd_data(data_dir=None, filename="cfd_data.xlsx", fit="linear"):
    """
    A very bespoke function to read heat transfer coefficients from an excel
    file

    Args:
        data_dir (str):
            Path to data file. The default is None. If unspecified the module
            liionpack.DATA_DIR folder will be used
        filename (str):
            The default is 'cfd_data.xlsx'.
        fit (str):
            options are 'linear' (default) and 'interpolated'.

    Returns:
        list:
            an interpolant is returned for each cell in the excel file.

    """
    if data_dir is None:
        data_dir = lp.DATA_DIR
    fpath = os.path.join(data_dir, filename)
    ncells = 32
    flow_bps = np.array(pd.read_excel(fpath, sheet_name="massflow_bps", header=None))
    temp_bps = np.array(pd.read_excel(fpath, sheet_name="temperature_bps", header=None))
    xv, yv = np.meshgrid(temp_bps, flow_bps)
    data = np.zeros([len(temp_bps), len(flow_bps), ncells])
    fits = []
    for i in range(ncells):
        data[:, :, i] = np.array(
            pd.read_excel(fpath, sheet_name="cell" + str(i + 1), header=None)
        )
        # funcs.append(interp2d(xv, yv, data[:, :, i], kind='linear'))
        if fit == "linear":
            fits.append(_fit_plane(xv, yv, data[:, :, i]))
        elif fit == "interpolated":
            fits.append(interp2d(xv, yv, data[:, :, i], kind="linear"))

    return data, xv, yv, fits


def get_linear_htc(planes, T, Q):
    """
    A very bespoke function that is called in the solve process to update the
    heat transfer coefficients for every battery - assuming linear relation
    between temperature, flow rate and heat transfer coefficient.

    Args:
        planes (list):
            each element of the list is a plane equation describing linear relation
            between temperature, flow rate and heat transfer coefficient.
        T (np.ndarray):
            The temperature of each battery.
        Q (float):
            The flow rate for the system.

    Returns:
        float:
            Heat transfer coefficient for each battery.

    """
    ncell = len(T)
    htc = np.zeros(ncell)
    for i in range(ncell):
        htc[i] = _z_from_plane(T[i], Q, planes[i])
    return htc


def get_interpolated_htc(funcs, T, Q):
    """
    A very bespoke function that is called in the solve process to update the
    heat transfer coefficients for every battery

    Args:
        funcs (list):
            each element of the list is an interpolant function.
        T (np.ndarray):
            The temperature of each battery.
        Q (float):
            The flow rate for the system.

    Returns:
        float:
            Heat transfer coefficient for each battery.

    """
    ncell = len(T)
    htc = np.zeros(ncell)
    for i in range(ncell):
        htc[i] = funcs[i](T[i], Q)
    return htc


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
        model (pybamm.lithium_ion.BaseModel):
            The PyBaMM model to solve with events added as variables.

    """
    for event in model.events:
        model.variables.update({"Event: " + event.name: event.expression})
    return model
