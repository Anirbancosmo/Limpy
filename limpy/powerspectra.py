d#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:15:59 2020

@author: anirbanroy
"""


from __future__ import division
import numpy as np
import limpy.params as p
import limpy.utils as lu

from powerbox import get_power

from scipy.interpolate import RectBivariateSpline, interp2d

from plotsettings import *


def powerspectra(x_grid, boxlength, ngrid, project_length, y_grid=None):
    
    cellsize=boxlength/ngrid
    
    nproj=project_length/cellsize
    
    if project_length is not None:
        
        if x_grid:
            g_xi, gx_j, x_ij = lu.slice(x_grid, ngrid, nproj, option='C')
        
        if y_grid is not None:
            g_yi, g_yj, y_ij = lu.slice(y_grid, ngrid, nproj, option='C')
            
        if y_grid is None:
            g_yi, g_yj, y_ij=g_xi, gx_j, x_ij
            
            
        X, Y=np.meshgrid(g_xi, g_xi)
            
        p_k_samples, bins_samples = get_power(samples, pb.boxlength,N=pb.N)  
    
    return 0


f1='/Users/anirbanroy/Documents/21cmFAST/Boxes/updated_smoothed_deltax_z007.60_512_80Mpc'
f1='/Users/anirbanroy/Documents/21cmFAST/Boxes/updated_smoothed_deltax_z007.60_512_80Mpc'
f3='/Users/anirbanroy/Documents/21cmFAST/Boxes/smoothed_deltax_z0.00_512_80Mpc'


def ps(fname, boxsize, opt="mean"):
    with open(fname, 'rb') as f:
            dens_gas = np.fromfile(f, dtype='f', count=-1)
            
    data=lu.slice_2d(dens_gas, 512, 10, operation=opt)
    
    data_mean=np.mean(data)
    
    d2p=data*data_mean+ data_mean
    
    p_k_m, bins_m = get_power(d2p, boxsize)
    
    return p_k_m, bins_m 
    
    
    
    
            
    
    
