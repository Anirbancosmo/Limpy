#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:55:59 2020

@author: anirbanroy
"""

"This is a dictionary of the default parameters"

# Parameters for SFR-LCII relation with scatter from Schaerer et. al 2020 
# arxiv: 2002.00979 (TABLE: A.1)
default_lcp_scatter_params={
       'a_off':6.98,
       'a_std':0.16,
       'b_off':0.99,
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
        'lcp_low': 1e3
        }


default_constants={
        'c_in_m': 3e8 #meter
        
        
        }
