# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:33:13 2021

@author: Tom
"""
from scipy.interpolate import interp1d, interp2d
import numpy as np
import pandas as pd
import os
import liionpack
from skspatial.objects import Plane
from skspatial.objects import Points

ROOT_DIR = os.path.dirname(os.path.abspath(liionpack.__file__))
CIRCUIT_DIR = os.path.join(ROOT_DIR, "circuits")
DATA_DIR = os.path.join(ROOT_DIR, "data")
INIT_FUNCS = os.path.join(ROOT_DIR, "init_funcs")


def interp_current(df):
    """
    Returns an interpolation function for current w.r.t time

    Args:
        df (pandas.DataFrame or Dict):
            Contains data for 'Time' and 'Cells Total Current' from which to
            construct an interpolant function

    Returns:
        f (function):
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
        funcs (list):
            an interpolant is returned for each cell in the excel file.

    """
    if data_dir is None:
        data_dir = liionpack.DATA_DIR
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
        htc (float):
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
        htc (float):
            Heat transfer coefficient for each battery.

    """
    ncell = len(T)
    htc = np.zeros(ncell)
    for i in range(ncell):
        htc[i] = funcs[i](T[i], Q)
    return htc


def build_inputs_dict(I_batt, htc):
    """
    Function to convert inputs and external_variable arrays to list of dicts
    As expected by the casadi solver. These are then converted back for mapped
    solving but stored individually on each returned solution.
    Can probably remove this process later

    Args:
        I_batt (np.ndarray):
            The input current for each battery.
        htc (np.ndarray):
            the heat transfer coefficient for each battery.

    Returns:
        inputs_dict (list):
            each element of the list is an inputs dictionary corresponding to each
            battery.


    """
    inputs_dict = []
    for i in range(len(I_batt)):
        inputs_dict.append(
            {
                # 'Volume-averaged cell temperature': T_batt[i],
                "Current function [A]": I_batt[i],
                "Total heat transfer coefficient [W.m-2.K-1]": htc[i],
            }
        )
    return inputs_dict
