#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:55:59 2020

@author: anirbanroy
"""

"""
This module is a collection of defaults parameters, cosmological paramneters 
and a list of flags which users need to mention to get their specific choices 
of output.

This file will be processed by the params.py file internally which will run the 
Limpy code to generate specific outputs
"""

"""
Cosmological Parameters
"""

line_params={
        
'line_name': 'CII'
        
        
        
        
        
        }

cosmo_params={
        
'h': 0.6766,
'omega_lambda': 0.6889,
'omgega_bh2': 0.02242,
'omgega_ch2': 0.11933,
'omega_mh2' : 0.14240,
'omega_k': -0.05,
'tau': 0.0561,   
'ns': 0.965

}


"""
Input parameters for output files
"""

code_params={
        'use_scatter': False,
        'use_pk': 'linear' # or nonlinear
    }

"""
Astrophysical Parameters
"""
astro_params={
        'Mmin': 1e9,
        'Mmax': 1e15,
        'Lsun': 3.828e26, #watts
        'delta_c':1.686,
        'halo_model': 'Tinker08'
        }


"""
Different line rest frame frequencies and other parameters
"""
default_line_params={
    'nu_CII': 1900, #GHz
    'nu_CO10': 115.27 #GHz
    }



"""
Different line rest frame frequencies and other parameters
"""
default_L_CII_scatter_params={
       'a_off':6.98,
       'a_std':0.16,
       'b_off':0.99,
       'b_std':0.09
       }


default_L_OIII_scatter_params={
	   'a_off':7.4,
       'a_std':0.16,
       'b_off':0.97,
       'b_std':0.09
}



# Parameters for SFR-LCII relation from Chung et. al 2020
default_L_CII_chung_params={
        'a': 1.4,
        'b': 0.07,
        'c': 7.1,
        'd': 0.07,
        }


# dummy values to extend the SFR-Mhalo relation when halomass goes below M_min
default_dummy_values={
        'lcp_low': 1e-12
        }


default_constants={
        'c_in_m': 3e8, #meter 
        'G_const': 6.67*1e-11,  # kg^{-2} m^2 
        
        'ghz_to_hz': 1e9,  # Giga-Hertz to Hertz
          
        # distances
        'mpc_to_m': 3.086e+22, # meter
        'm_to_mpc': 3.24e-23, # Mpc
        
        'km_to_m': 1e3, #meter
        'minute_to_degree': 1.0/60,
        'degree_to_minute': 60.0,
        'GHz_to_Hz': 1e9,
        'Jy': 1e-26, # Watts. m^{-2} HZ^{-1}
        'kb_si': 1.38e-23 #J/K
    
        
        }


experiments={'ccatp':
        {
                220:{'nu_central':220,'dnu':2.2, 'theta': 57},
                280:{'nu_central':280,'dnu':2.8, 'theta': 45},
                350:{'nu_central':350,'dnu':3.5, 'theta': 35},
                410:{'nu_central':410,'dnu':4.1, 'theta': 30},
        }
        }