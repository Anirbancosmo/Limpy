#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:55:51 2020

@author: anirbanroy
"""

import inputs as inp
import cosmos
import numpy as np

#parameters

line_name=inp.line_params['line_name']

Mmin=inp.astro_params['Mmin']
Mmax=inp.astro_params['Mmax']

Lsun=inp.astro_params['Lsun']
delta_c=inp.astro_params['delta_c']
Halo_model=inp.astro_params['halo_model']


small_h=inp.cosmo_params['h']
omega_matter=inp.cosmo_params['omega_mh2']/small_h**2
omega_lambda=inp.cosmo_params['omega_lambda']

use_scatter=inp.code_params['use_scatter']

def line_scattered_params(line_name='CII'):
    if(line_name=='CII'):
        a_off=inp.default_L_CII_scatter_params['a_off']
        a_std=inp.default_L_CII_scatter_params['a_std']
        b_off=inp.default_L_CII_scatter_params['b_off']
        b_std=inp.default_L_CII_scatter_params['b_std']
    if(line_name=='OIII'):
        a_off=inp.default_L_OIII_scatter_params['a_off']
        a_std=inp.default_L_OIII_scatter_params['a_std']
        b_off=inp.default_L_OIII_scatter_params['b_off']
        b_std=inp.default_L_OIII_scatter_params['b_std']
        
    return a_off, a_std,  b_off,  b_std
    


def nu_rest(line_name='CII'):
    if(line_name=='CII'):
        nu=1900
    
    if(line_name=='OIII'):
        nu=3400
    
    if(line_name[0:2]=='CO'):
        line_name_len=len(line_name)
        if(line_name_len==4):
            J_lader=int(line_name[2])
        elif(line_name_len==5 or line_name_len==6):
            J_lader=int(line_name[2:4])
        nu=J_lader*115.27
        
    return nu



lcp_low=inp.default_dummy_values['lcp_low']

cosmo=cosmos.cosmo()

"""
# IF we want to keep same cosmological parameters for the code and HMFcal
"""
#from hmf import cosmo as cosmo_hmf
#my_cosmo = cosmo_hmf.Cosmology()
#my_cosmo.update(cosmo_params={"H0":71,"Om0":0.281,"Ode0":0.719,"Ob0":0.046})



#Constants
c_in_m= 3e8 #meter/s
c_in_mpc=9.72e-15 #Mpc/s
mpc_to_m= 3.086e+22 # meter
m_to_mpc= 3.24e-23 # Mpc
km_to_m= 1e3 #meter
jy_unit= 1e-26 # Watts. m^{-2} HZ^{-1}
Ghz_to_hz= 1e9  # Giga-Hertz to Hertz
kb_si= 1.38e-23 #J/K

minute_to_degree: 1.0/60
degree_to_minute: 60.0

nu_rest_CII=1900 #GHZ
nu_rest_CO10=115 #GHZ

degree_to_minute=60.0
minute_to_degree=1.0/60
minute_to_radian=(np.pi/10800)

degree_to_radian=(np.pi/180)
