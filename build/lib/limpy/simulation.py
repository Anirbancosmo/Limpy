#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:56:15 2020

@author: anirbanroy
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
#from plotsettings import *

sfr_filepath='../data/'
f=np.loadtxt(sfr_filepath+'sfr_beherozzi.dat')
z=f[:,0]
m=f[:,1]
sfr=f[:,2]
zlen=137 #manually checked 
mlen=int(len(z)/zlen)
zn=z[0:zlen]
mhn=m.reshape(mlen,zlen)[:,0]
sfrn=sfr.reshape(mlen,zlen)
sfr_interpolation=RectBivariateSpline(mhn, zn, sfrn)

def sfr_int(m,z):
    res=sfr_interpolation(m,z) 
    
    res=np.where(res<1e-4, 0.0, res)
    return res.flatten()

 
