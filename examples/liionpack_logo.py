# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 21:19:38 2021

@author: tom
"""

from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import liionpack as lp
import pybamm
import os

im = imread("liionpack.PNG")
im = resize(im[:, :, 0], (im.shape[0] // 8, im.shape[1] // 8),
                       anti_aliasing=True)
plt.figure()
plt.imshow(im)

Ns, Np = im.shape


netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=5e-5,
                           Rc=1e-2, Ri=1e-2, V=3.6, I=20,
                           terminals="left-right")
parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
experiment = pybamm.Experiment(
    operating_conditions=[f"Discharge at {2*Np} A for 10 minutes"], period="1 minute"
)

htc = im.T.flatten()
htc = 1.5 - htc
htc *= 5

output = lp.solve(
    netlist=netlist,
    parameter_values=parameter_values,
    experiment=experiment,
    sim_func=lp.thermal_simulation,
    output_variables=["Volume-averaged cell temperature [K]"],
    inputs={"Total heat transfer coefficient [W.m-2.K-1]": htc},
    nproc=os.cpu_count()
)

data = output["Volume-averaged cell temperature [K]"][-1, :]
lp.plot_cell_data_image(netlist, data, tick_labels=False, figsize=(15, 6))