#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:55:51 2020

@author: anirbanroy
"""
import numpy as np
import inputs as inp
import cosmos

#parameters
Mmin=inp.astro_params['Mmin']
Mmax=inp.astro_params['Mmax']

Lsun=inp.astro_params['Lsun']
delta_c=inp.astro_params['delta_c']
Halo_model=inp.astro_params['halo_model']

c_in_m=inp.default_constants['c_in_m']
mpc_to_m=inp.default_constants['mpc_to_m']
m_to_mpc=inp.default_constants['m_to_mpc']
jy_unit=inp.default_constants['Jy']
Ghz_to_hz=inp.default_constants['ghz_to_hz']
kb_si=inp.default_constants['kb_si']

nu_rest_line=inp.line_frequency['nu_CII']
nu_rest_CO10=inp.line_frequency['nu_CO10']

small_h=inp.cosmo_params['h']
omega_matter=inp.cosmo_params['omega_mh2']/small_h**2
omega_lambda=inp.cosmo_params['omega_lambda']

use_scatter=inp.code_params['use_scatter']


a_off=inp.default_lcp_scatter_params['a_off']
a_std=inp.default_lcp_scatter_params['a_std']
b_off=inp.default_lcp_scatter_params['b_off']
b_std=inp.default_lcp_scatter_params['b_std']

cosmo=cosmos.cosmo()

