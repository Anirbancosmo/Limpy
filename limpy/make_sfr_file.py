#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:00:41 2020
@author: anirbanroy
"""
import numpy as np
from scipy.interpolate import interp2d, RectBivariateSpline, RegularGridInterpolator, SmoothBivariateSpline
import matplotlib.pyplot as plt  
from plotsettings import *


f=np.loadtxt('/Users/anirbanroy/Documents/Limpy/data/sfh_z0_z8/sfr/sfr_release.dat')

z=f[:,0]-1
mh=(f[:,1])
sfr=(f[:,2])
sfr=np.where(sfr==-1000, 1e-10, sfr)

fi=open("sfr_beherozzi.dat", 'w+')
fi.write("#z\t\t\tM_halo \t\t\t sfr\n")
for i in range(len(z)):
    if(sfr[i]==1e-10):
        sfr[i]=1e-10
        
    else:
        sfr[i]=10**sfr[i]
    fi.write("%e\t\t%e\t\t%e\n" %(z[i], 10**mh[i], sfr[i]))
    
fi.close()



f=np.loadtxt('/Users/anirbanroy/Documents/Limpy/limpy/sfr_beherozzi.dat')
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

 
