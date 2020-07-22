#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:55:59 2020

@author: anirbanroy
"""

"This is a dictionary of the default parameters"

# Parameters for SFR-LCII relation with scatter from Schaerer et. al 2020 
# arxiv: 2002.00979 (TABLE: A.1)

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

code_params={
        'use_scatter': False,
        'use_pk': 'linear' # or nonlinear
    }


astro_params={
        'Mmin': 1e11,
        'Mmax': 1e15,
        'Lsun': 3.828e26, #watts
        'delta_c':1.686,
        'halo_model': 'Tinker08'
        }


line_frequency={
    'nu_CII': 1900 #GHz
    }


default_lcp_scatter_params={
       'a_off':6.98,
       'a_std':0.16,
       'b_off':0.99,
       'b_std':0.09
       }


default_lot_scatter_params={
	   'a_off':7.4,
       'a_std':0.16,
       'b_off':0.97,
       'b_std':0.09
}



# Parameters for SFR-LCII relation from Chung et. al 2020
default_lcp_chung_params={
        'a': 1.4,
        'b': 0.07,
        'c': 7.1,
        'd': 0.07,
        }


# dummy values to extend the SFR-Mhalo relation when halomass goes below M_min
default_dummy_values={
        'log_lcp_low': 0.0
        }


default_constants={
        'c_in_m': 3e8, #meter 
        'G_const': 6.67*1e-11,  # kg^{-2} m^2 
        
        
        # distances
        'mpc_to_m': 3.086e+22, # meter
        'm_to_mpc': 3.24e-23, # Mpc
        
        'km_to_m': 1e3, #meter
        'minute_to_degree': 1.0/60,
        'degree_to_minute': 60.0,
        'GHz_to_Hz': 1e9,
        'Jy': 1e-26 # Watts. m^{-2} HZ^{-1}
        
        }


experiments={'ccatp':
        {
                220:{'nu_central':220,'dnu':2.2, 'theta': 57},
                280:{'nu_central':280,'dnu':2.8, 'theta': 45},
                350:{'nu_central':350,'dnu':3.5, 'theta': 35},
                410:{'nu_central':410,'dnu':4.1, 'theta': 30},
        }
        }