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

line_params = {"line_name": "CII158"}

cosmo_params = {
    "h": 0.68,
    "omega_lambda": 0.70,
    "omgega_bh2": 0.02226,
    "omgega_ch2": 0.11933,
    "omega_mh2": 0.1433,
    "omega_k": 0.0,
    "tau": 0.054,
    "ns": 0.97,
    "sigma8": 0.81,
}


"""
Input parameters for output files
"""

code_params = {"use_scatter": False, "use_pk": "linear"}  # or nonlinear

"""
Astrophysical Parameters
"""
astro_params = {
    "Mmin": 1e10,
    "Mmax": 1e19,
    "Lsun": 3.828e26,  # watts
    "delta_c": 1.68,
    "halo_model": "sheth99",
    "halo_mass_def": "fof",
    "bias_model": "sheth01",
    "bias_mass_def": "vir"
}


"""
Different line rest frame frequencies and other parameters
"""
default_L_CII_scatter_params = {
    "a_off": 6.98,
    "a_std": 0.16,
    "b_off": 0.99,
    "b_std": 0.09,
}


default_L_OIII_scatter_params = {
    "a_off": 7.4,
    "a_std": 0.16,
    "b_off": 0.97,
    "b_std": 0.09,
}


# Parameters for SFR-LCII relation from Chung et. al 2020
default_L_CII_chung_params = {
    "a": 1.4,
    "b": 0.07,
    "c": 7.1,
    "d": 0.07,
}

# Parameters for SFR-LCO relation per transition from Greve et. al 2014

default_L_CO_1_0_scatter_params = {
    "a_off": 0.99,
    "a_std": 0.04,
    "b_off": 1.90,
    "b_std": 0.40,
}

default_L_CO_2_1_scatter_params = {
    "a_off": 1.03,
    "a_std": 0.09,
    "b_off": 1.60,
    "b_std": 0.90,
}

default_L_CO_3_2_scatter_params = {
    "a_off": 0.99,
    "a_std": 0.04,
    "b_off": 2.10,
    "b_std": 0.04,
}

default_L_CO_4_3_scatter_params = {
    "a_off": 1.08,
    "a_std": 0.09,
    "b_off": 1.20,
    "b_std": 0.90,
}

default_L_CO_5_4_scatter_params = {
    "a_off": 0.97,
    "a_std": 0.06,
    "b_off": 2.50,
    "b_std": 0.60,
}

default_L_CO_6_5_scatter_params = {
    "a_off": 0.93,
    "a_std": 0.05,
    "b_off": 3.10,
    "b_std": 0.50,
}

default_L_CO_7_6_scatter_params = {
    "a_off": 0.87,
    "a_std": 0.05,
    "b_off": 3.90,
    "b_std": 0.40,
}

default_L_CO_8_7_scatter_params = {
    "a_off": 0.66,
    "a_std": 0.07,
    "b_off": 5.80,
    "b_std": 0.60,
}

default_L_CO_9_8_scatter_params = {
    "a_off": 0.82,
    "a_std": 0.07,
    "b_off": 4.60,
    "b_std": 0.60,
}

default_L_CO_10_9_scatter_params = {
    "a_off": 0.66,
    "a_std": 0.07,
    "b_off": 6.10,
    "b_std": 0.60,
}

default_L_CO_11_10_scatter_params = {
    "a_off": 0.57,
    "a_std": 0.09,
    "b_off": 6.80,
    "b_std": 0.70,
}


# dummy values to extend the SFR-Mhalo relation when halomass goes below M_min
default_dummy_values = {"lcp_low": 1e-12}


default_constants = {
    "c_in_m": 3e8,  # meter
    "G_const": 6.67 * 1e-11,  # kg^{-2} m^2
    "ghz_to_hz": 1e9,  # Giga-Hertz to Hertz
    # distances
    "mpc_to_m": 3.086e22,  # meter
    "m_to_mpc": 3.24e-23,  # Mpc
    "c_in_mpc": 9.72e-15,  # Mpc/s
    "km_to_mpc": 3.2408e-20,
    "km_to_m": 1e3,  # meter
    "minute_to_degree": 1.0 / 60,
    "degree_to_minute": 60.0,
    "GHz_to_Hz": 1e9,
    "Jy": 1e-26,  # Watts. m^{-2} HZ^{-1}
    "kb_si": 1.38e-23,  # J/K
}
