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


import powerbox as pbox

boxsize=80
ndim=512

pb = pbox.PowerBox(
    N=ndim,                     # Number of grid-points in the box
    dim=2,                     # 2D box
    pk = lambda k: 0.1*k**-2., # The power-spectrum
    boxlength = boxsize,           # Size of the box (sets the units of k in pk)
    seed = 1010                # Set a seed to ensure the box looks the same every time (optional)
)

#plt.imshow(pb.delta_x(),extent=(0,1,0,1))
#plt.colorbar()
#plt.show()


pb = pbox.PowerBox(
    N=ndim,                     # Number of grid-points in the box
    dim=2,                     # 2D box
    pk = lambda k: 0.1*k**-2., # The power-spectrum
    boxlength = boxsize,           # Size of the box (sets the units of k in pk)
    seed = 1010,               # Set a seed to ensure the box looks the same every time (optional)
    ensure_physical=True       # ** Ensure the delta_x is a physically valid over-density **
)


#plt.imshow(pb.delta_x(),extent=(0,1,0,1))
#plt.colorbar()
#plt.show()


lnpb = pbox.LogNormalPowerBox(
    N=ndim,                     # Number of grid-points in the box
    dim=2,                     # 2D box
    pk = lambda k: 0.1*k**-2., # The power-spectrum
    boxlength = boxsize,           # Size of the box (sets the units of k in pk)
    seed = 1010                # Use the same seed as our powerbox
)
#plt.imshow(lnpb.delta_x(),extent=(0,1,0,1))
#plt.colorbar()
#plt.show()


# Create a discrete sample using the PowerBox instance.
samples = pb.create_discrete_sample(nbar=50000,      # nbar specifies the number density
                                    min_at_zero=True  # by default the samples are centred at 0. This shifts them to be positive.
                                   )
ln_samples = lnpb.create_discrete_sample(nbar=50000, min_at_zero=True)




p_k_field, bins_field = get_power(pb.delta_x(), pb.boxlength)
p_k_lnfield, bins_lnfield = get_power(lnpb.delta_x(), lnpb.boxlength)

# The number of grid points are also required when passing the samples
p_k_samples, bins_samples = get_power(samples, pb.boxlength,N=pb.N)
p_k_lnsamples, bins_lnsamples = get_power(ln_samples, lnpb.boxlength,N=lnpb.N)