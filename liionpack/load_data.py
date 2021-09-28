# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:29:24 2021

@author: Tom
"""
import pandas as pd
import os
from liionpack import batteryrig_utils

def load_simulation_data(data_dir, filename='A25022021_2_D'):
    r'''
    A bespoke utility function to load in experimental data

    Parameters
    ----------
    data_dir : str
        The parent directory for the data file.
    filename : str, optional
        The filename for the data. The default is 'A25022021_2_D'.

    Returns
    -------
    df : pandas.Dataframe
        An experimental data file

    '''
    df = pd.read_csv(os.path.join(data_dir, filename + '.txt'))
    # calculate zero currents
    df = batteryrig_utils.zero_currents(df)
    df = batteryrig_utils.calc_cell_voltages(df)
    st = 172
    tmap = df['Time'] > st
    df = df[tmap]
    tlow = df['Time'].min()
    df['Time'] -= tlow
    df.set_index('Time',inplace=True,drop=False)
    return df