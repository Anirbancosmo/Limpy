#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 02:38:29 2020

@author: anirbanroy
"""


import matplotlib.pyplot as plt
import numpy as np

from powerbox import get_power
from multiprocessing import cpu_count

import lline as ll


import utils

boxsize=80
halo_redshift=7.6
nu_obs=220
dnu=2.2
line_name='CII'
ndim_original=3

mmin=1e10
ngrid=512


halo_file='/Users/anirbanroy/Documents/21cmFAST/Output_files/Halo_lists/halos_z7.60_512_80Mpc'

proj_L=utils.length_projection(nu_obs=nu_obs, dnu=dnu, line_name=line_name)

hm, cm=utils.make_halocat(halo_file, filetype='dat',boxsize=boxsize)

mass_cut=hm >= mmin
halomass_cut=hm[mass_cut]

print("Calculating line luminosities")
lum_line=np.zeros(len(halomass_cut))
for i in range(len(halomass_cut)):
    lum_line[i]=ll.mhalo_to_lline(halomass_cut[i],halo_redshift,line_name=line_name)

print("making the luminsoity grid")
gi=ll.calc_intensity_3d(boxsize, ngrid, halo_file,halo_redshift, line_name='CII',\
                        halo_cutoff_mass=1e11, use_scatter=False,halocat_file_type='dat', unit='mpc')

print("Calculating power spectra")

k, pk= utils.powerspectra_2d(gi, boxsize, ngrid, project_length=proj_L, volume_normalization=False)

