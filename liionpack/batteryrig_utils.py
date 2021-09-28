# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:57:53 2021

@author: dominicdathan
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def zero_currents(df):
    
    # zero currents
    offset_time = 10
    idx = df[df['Time']<offset_time].index[-1]
    # list of labels to sum currents
    current_cols = []
    for i in range(1,17):
        col_name = 'CellCurrent_'+str(i)+'_'+str(i+16)
        col_name_zero = 'CellCurrent_'+str(i)+'_'+str(i+16) + '_zeroed'
        offset = df[col_name].iloc[:idx+1].mean()
        df[col_name_zero] = df[col_name] - offset
        current_cols.append(col_name_zero)
        
    # total current
    df['Cells Total Current'] = df[current_cols].sum(axis=1)
        
    return df

def calc_cell_voltages(df):
    for i in range(1,17):
        df['CellVoltage_' + str(i)] = df['TerminalVoltage'] - df['CellVoltage_' + str(i)+'_'+str(i+16)]
        df['CellVoltage_' + str(i+16)] = df['CellVoltage_' + str(i)+'_'+str(i+16)]
    return df

def get_list_of_cols(colstr):
    cols = []
    
    if colstr == 'temperature':
        for i in range(1,33):
            cols.append('CellTemp_' + str(i))
    elif colstr == 'voltage':
        for i in range(1,33):
            cols.append('CellVoltage_' + str(i))
    elif colstr == 'current':
        for i in range(1,17):
            cols.append('CellCurrent_'+str(i)+'_'+str(i+16))
        cols.extend(cols)
    return cols

def cell_XY_positions():
    # Cell position data
    X_pos = [0.080052414,0.057192637,0.080052401,0.057192662,0.080052171,0.057192208,0.080052285,0.057192264,
    -0.034260006,-0.011396764,-0.034259762,-0.011396799,-0.034259656,-0.011397055,-0.034259716,-0.01139668,
    0.034329391,0.01146636,0.034329389,0.011466487,0.034329301,0.011466305,0.034329448,0.011465906,
    -0.079983086,-0.057122698,-0.079983176,-0.057123076,-0.079982958,-0.057122401,-0.079982995,-0.057122961]
    
    Y_pos = [-0.046199913,-0.033000108,-0.019799939,-0.0066001454,0.0066000483,0.019799888,0.033000056,0.046200369,
    0.046200056,0.033000127,0.019800097,0.0065999294,-0.0065998979,-0.019800061,-0.032999967,-0.046200222,
    -0.04620005,-0.032999882,-0.019800016,-0.0065999624,0.0065997543,0.019799885,0.033000077,0.046199929,
    0.0462001,0.033000148,0.019800099,0.0066000627,-0.0065999586,-0.019800142,-0.032999927,-0.046199973]
    return X_pos,Y_pos

def plot_time_vertical(ax,t):
    y_min, y_max = ax.get_ylim()
    ax.plot([t,t],[y_min, y_max],"k--")

def cell_text(ax,vals,prec,text_colors):
    X_pos, Y_pos = cell_XY_positions()
    for i, val in enumerate(vals):
        ax.text(x=X_pos[i],y=Y_pos[i],s="{:.{}f}".format(val,prec),color=text_colors[i],ha='center',va='center')

def text_color(vals,vmin,vmax,cmap):
    # return list of either black or white to write text, depending on whether plotted color is closer to white or black
    cm = mpl.cm.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    val_cm = cm(norm(vals))[:,:3]
    val_norm = np.dot(val_cm, [0.2989, 0.5870, 0.1140])
    return ['k' if v>0.5 else 'w' for v in val_norm]
    
def cell_text_numbers(ax,text_colors):
    X_pos, Y_pos = cell_XY_positions()
    y_offset = 0.005
    for i, [x_pos, y_pos] in enumerate(zip(X_pos, Y_pos)):
        ax.text(x=x_pos, y=y_pos - y_offset, s="{:d}".format(i+1), color=text_colors[i], ha='center', va='top',fontsize=7)
    
def cell_batch_color(ax,s):
    X_pos, Y_pos = cell_XY_positions()
    batch_1 = [1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32]
    batch_1 = [x-1 for x in batch_1]
    batch_2 = [3,7,11,15,19,23,27,31]
    batch_2 = [x-1 for x in batch_2]
    # batch 2
    ax.scatter([X_pos[i] for i in batch_2],[Y_pos[i] for i in batch_2],s=s, edgecolors='r', facecolors='none')
    # batch 1
    ax.scatter([X_pos[i] for i in batch_1],[Y_pos[i] for i in batch_1],s=s, edgecolors='k', facecolors='none')

def cell_scatter_plot(ax, c, val_text=False,  cellno_text=False, batch_col=False, text_prec=1, **kwargs):
    """

    Parameters
    ----------
    ax : matplotlib axis obj
        axis to plot on.
    c : like plt.scatter c kwarg
        colors to plot scatter with.
    val_text : bool
        write values on plot
    cellno_text : bool
        write cell numbers on plot
    batch_col :
        color cells by batch
    text_prec : int
        precsition to write text of values on cells.
    **kwargs : 
        plt.scatter kwargs.

    Returns
    -------
    None.

    """
    
    X_pos, Y_pos = cell_XY_positions()
    
    # set size of markers
    diameter = 21.44/1000;
    area = np.pi*(diameter/2)**2;
    s = area*4/np.pi # points 
    s = s*72/0.0254*1000 # some scaling... 
    
    # scatter plot
    sc = ax.scatter(X_pos, Y_pos, s, c, **kwargs)
    # set limits
    ax.set_xlim(-0.1,0.1)
    ax.set_ylim(-0.06,0.06)
    # colorbar
    plt.colorbar(sc, ax = ax, orientation = 'vertical')
    # set axis equal
    ax.set_aspect('equal')
    ax.set_axis_off()
    
    vmin, vmax = sc.get_clim()
    if 'cmap' in kwargs:
        cmap = kwargs.get('cmap')
    else:
        cmap='viridis'

    text_colors =  text_color(c,vmin,vmax,cmap)
    # write cell text
    if val_text:
        cell_text(ax,c,text_prec,text_colors)
    if cellno_text:        
        cell_text_numbers(ax, text_colors)
    if batch_col:
        cell_batch_color(ax,s)
    
def plot_var_list(var_list,file,data_dir=r'C:\D2HGROUP\D2H Group\092-002-AMMBa - Incoming\Data\oversampled'):
    if 'oversampled' in data_dir:
        df = pd.read_csv(os.path.join(data_dir, file + '.txt'))
    else:
        df = pd.read_csv(os.path.join(data_dir, file + '.txt'), skiprows=[1])
        df = df.iloc[1::100]
    df = zero_currents(df)
    df = calc_cell_voltages(df)
    data=[]
    for var in var_list:
        trace = go.Scatter(x=df['Time'],y=df[var],
                           mode='lines',
                           name=var)
        data.append(trace)
    fig =go.Figure(data=data)
    fig.show()
    
    
def oversampling(file_name:str, directory:str, N:int=100, derivative_limit:float=10, adaptive:bool=True):
    """
    Give a file for the 2021 AMMBa battery testing a new file under an
    oversampling directory is created. End tag of _ad signifies adaptive
    oversampling was used.
    Parameters
    ----------
    file_name : str
        File name (with extension) to oversample found in the specified
        directory.
    directory : str
        Directory where the specified file can be found. An oversampling folder
        will be created here.
    N : int, optional
        Number of points to average together. The default is 100.
    derivative_limit : float, optional
        Limit at which oversampling is conducted. A current derivative over
        this value will result in all data within N/2 points to be kept.
        The default is 10.
    adaptive : bool, optional
        If adaptive sampling is used. Without all data will be over sampled
        and high derivative points may be missed. The default is True

    Returns
    -------
    None.

    """
    oversampled_folder = os.path.join(directory, 'oversampled')
    if not os.path.isdir(oversampled_folder):
        os.mkdir(oversampled_folder)
        
        
    look_forward = int((N-1)/2)
    look_back = look_forward + ((N+1) % 2)
    
    df = pd.read_csv(os.path.join(directory, file_name), skiprows=[1])
    n_points = len(df)
    
    if not adaptive:
        ii = 0
    
        idx = list(range(ii,look_forward+1))
        df_temp = pd.DataFrame(df.iloc[idx].mean(axis=0))
        df_temp = df_temp.transpose()
        df_temp['Time'] = df['Time'].iloc[ii]
        df_new = pd.DataFrame(data=df_temp, columns=df.columns)
        ii += N
        done = False
        while not done:
            if ii+look_forward > n_points-1:
                ee = n_points-1
                done = True
            else:
                ee = ii+look_forward
            idx = list(range(ii-look_back, ee+1))
        
            df_temp = pd.DataFrame(df.iloc[idx].mean(axis=0))
            df_temp = df_temp.transpose()
            df_temp['Time'] = df['Time'].iloc[ii]
            df_new = df_new.append(df_temp, ignore_index=True)
            
            ii += N
            if ii >= n_points:
                done = True
                
    else:
        time = np.array(df['Time'])
        dT = np.append(time[1:] - time[:-1],1)
        current = sum(np.transpose(np.array(df.iloc[:,52:68])))
        dI = np.append(current[1:] - current[:-1],0)
        df = df[dT != 0]
        dI = dI[dT != 0]
        dT = dT[dT != 0]        
        
        derivative = abs(dI/dT)
        n_points = len(dI)
        
        save_points = np.full(n_points, False, dtype=bool)
        for i in range(n_points):
            if derivative[i] > derivative_limit:
                if i-look_back < 0:
                    bb = 0
                else:
                    bb = i-look_back
                    
                if i+look_forward > n_points-1:
                    ff = n_points-1
                else:
                    ff = i+look_forward
                idx = list(range(bb, ff))
                
                save_points[idx] = True
                
        
        df_new = pd.DataFrame(data = df.iloc[0], columns=df.columns)
        idx = []
        for i in range(1,n_points-1):
            if save_points[i]:
                idx = [i]
            else:
                if len(idx) == 0:
                    idx = [i]
                else:
                    idx.append(i)
                    
            if (len(idx) == N) or (save_points[i+1] or (save_points[i])):
                df_temp = pd.DataFrame(df.iloc[idx].mean(axis=0))
                df_temp = df_temp.transpose()
                df_new = df_new.append(df_temp, ignore_index=True)
                
                idx = []

                
    df_new['Time'] = df_new['Time'] - df_new['Time'][0]
    output_file = os.path.join(oversampled_folder, file_name)
    df_new.to_csv(output_file, sep=',', index=False, float_format='%.6g')
    
    return
    
def cell_HTC(file_dir:str, file_name:str, start_time:float,
             C:float=52.5, A:float=4.7688e-3, skiprows=None,
             save_loc:str='', auto_open:bool=True):
    
    input_file = os.path.join(file_dir,file_name+'.txt')
    df = pd.read_csv(input_file,skiprows=skiprows)
    
    cell_temp = ['CellTemp_{}'.format(i) for i in range(1,33)]
    air_temp = 'AirTemp_1'
    df = df[df['Time'] > start_time].reset_index(drop=True)
    df['Time'] = df['Time'] - df['Time'][0]
    
    n_points = len(df['Time'])
    deltaT0 = [ df[cell][0]-df[air_temp][0] for cell in cell_temp]
    deltaT  = [[df[cell][i]-df[air_temp][i] for i in range(n_points)] for cell in cell_temp]
    
    lhs = np.array([[np.log(deltaT[j][i]/deltaT0[j]) for i in range(n_points)] for j in range(len(cell_temp))])
    rhs = np.array(A*df['Time']/C)
    
    fig = make_subplots(x_title='-tA/C')
    for i,cell in enumerate(cell_temp):
        fig.add_trace(go.Scatter(x=rhs, y=lhs[i][:], mode='lines',name=cell_temp[i]))
    
    fig.update_yaxes(title_text='ln(deltaT/deltaT0)')
    layout = go.Layout(title=file_name)
    fig.update_layout(layout)
    fig.show()
    fig.write_html(os.path.join(save_loc,file_name+'.html'), auto_open=auto_open)
    
    # HTC calc
    htc = []
    mDot_idx = []
    for i in range(len(cell_temp)):
        idx = np.array(deltaT[i]) > 1.5
        x = np.array(rhs[idx])
        y = np.array(lhs[i][idx])
        
        n = len(x)
        xyi = sum([ x[i]*y[i] for i in range(n)])
        xi = sum(x)
        yi = sum(y)
        xi2 = sum([x[i]**2 for i in range(n)])
        
        htc.append(-(n*xyi-xi*yi)/(n*xi2-xi**2))
        
        if (len(mDot_idx) == 0) or (len(idx) < len(mDot_idx)):
            mDot_idx = idx
    
    htc.append(np.mean(df['MassFlowRate'][mDot_idx]))
    cell_temp.append('MassFlowRate')
    htcS = pd.Series(htc, index=cell_temp)
    htcS.to_csv(os.path.join(save_loc,file_name+'.txt'),header=False)    
    # print(htc)
    return
