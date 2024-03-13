#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:55:59 2020

@author: anirbanroy
"""

"""
This module is a collection of defaults parameters, cosmological parameters 
and a list of flags which users need to mention to get their specific choices 
of output.

This file will be processed by the params.py file internally which will run the 
Limpy code to generate specific outputs
"""

"""
Cosmological Parameters
"""

line_name = "CII158"

# Cosmological parameters
h = 0.6776
omega_lambda = 0.6889
omega_b = 0.0489
omega_m =  0.3111
omega_k = 0.0
tau = 0.056
ns = 0.9665
sigma_8 = 0.81

"""
Input parameters for output files
"""

use_scatter = False
use_pk = "linear"  # or nonlinear


"""
Line parameters
"""

use_scatter=False
a_off = 6.98
a_std = 0.16
b_off = 0.99
b_std= 0.09
f_duty=0.1
zg=8.8
scatter_dex = 0.2
cov_scatter_matrix=None

"""
Astrophysical Parameters
"""

M_min = 1e8
M_max = 1e19
delta_c = 1.68
halo_model = "sheth99"
halo_mass_def = "fof"
bias_model = "sheth01"
bias_mass_def = "vir"

"""
Different line rest frame frequencies and other parameters
"""

# Parameters for CII line scattering
a_off_CII_158 = 6.98
a_std_CII_158 = 0.16
b_off_CII_158 = 0.99
b_std_CII_158 = 0.09

# Parameters for OIII line scattering
a_off_OIII_88 = 7.4
a_std_OIII_88 = 0.16
b_off_OIII_88 = 0.97
b_std_OIII_88 = 0.09

# Parameters for SFR-LCII relation from Chung et. al 2020
a_CII = 1.4
b_CII = 0.07
c_CII = 7.1
d_CII = 0.07

# Parameters for SFR-LCO relation per transition from Greve et. al 2014
a_off_CO_1_0 = 0.99
a_std_CO_1_0 = 0.04
b_off_CO_1_0 = 1.90
b_std_CO_1_0 = 0.40

a_off_CO_2_1 = 1.03
a_std_CO_2_1 = 0.09
b_off_CO_2_1 = 1.60
b_std_CO_2_1 = 0.90

a_off_CO_3_2 = 0.99
a_std_CO_3_2 = 0.04
b_off_CO_3_2 = 2.10
b_std_CO_3_2 = 0.04

a_off_CO_4_3 = 1.08
a_std_CO_4_3 = 0.09
b_off_CO_4_3 = 1.20
b_std_CO_4_3 = 0.90

a_off_CO_5_4 = 0.97
a_std_CO_5_4 = 0.06
b_off_CO_5_4 = 2.50
b_std_CO_5_4 = 0.60

a_off_CO_6_5 = 0.93
a_std_CO_6_5 = 0.05
b_off_CO_6_5 = 3.10
b_std_CO_6_5 = 0.50

a_off_CO_7_6 = 0.87
a_std_CO_7_6 = 0.05
b_off_CO_7_6 = 3.90
b_std_CO_7_6 = 0.40

a_off_CO_8_7 = 0.66
a_std_CO_8_7 = 0.07
b_off_CO_8_7 = 5.80
b_std_CO_8_7 = 0.60

a_off_CO_9_8 = 0.82
a_std_CO_9_8 = 0.07
b_off_CO_9_8 = 4.60
b_std_CO_9_8 = 0.60

a_off_CO_10_9 = 0.66
a_std_CO_10_9 = 0.07
b_off_CO_10_9 = 6.10
b_std_CO_10_9 = 0.60

a_off_CO_11_10 = 0.57
a_std_CO_11_10 = 0.09
b_off_CO_11_10 = 6.80
b_std_CO_11_10 = 0.70

a_off_CO_12_11 = 0.51
a_std_CO_12_11 = 0.11
b_off_CO_12_11 = 7.50
b_std_CO_12_11 = 0.80

a_off_CO_13_12 = 0.47
a_std_CO_13_12 = 0.20
b_off_CO_13_12 = 7.90
b_std_CO_13_12 = 1.50

# Dummy values to extend the SFR-Mhalo relation when halomass goes below M_min
lcp_low = 1e-12

# Constants
c_in_m = 3e8  # meter
G_const = 6.67 * 1e-11  # kg^{-2} m^2
ghz_to_hz = 1e9  # Giga-Hertz to Hertz

Lsun = 3.828e26  # watts


# Distances
mpc_to_m = 3.086e22  # meter
m_to_mpc = 3.24e-23  # Mpc
c_in_mpc = 9.72e-15  # Mpc/s
km_to_mpc = 3.2408e-20
km_to_m = 1e3  # meter
minute_to_degree = 1.0 / 60
degree_to_minute = 60.0
GHz_to_Hz = 1e9
Jy = 1e-26  # Watts. m^{-2} HZ^{-1}
kb_si = 1.38e-23  # J/K

Lsun_erg_s = 3.828 * 1e33  # rg/s
minute_to_degree: 1.0 / 60
degree_to_minute: 60.0
nu_rest_CII = 1900  # GHZ
nu_rest_CO10 = 115  # GHZ
degree_to_minute = 60.0
minute_to_degree = 1.0 / 60
minute_to_radian = 0.000290888208
degree_to_radian = 0.0174533

################################################################################
# Some Functions based on input parameters
###############################################################################

def freq_to_lambda(nu):
    "frequency to wavelength converter."
    wav = c_in_mpc / nu
    return wav  # in mpc/h unit


def lambda_line(line_name="CI371"):
    """
    Wavelngth of the line.

    parameters
    ----------
    line_name: str
            name of the line to calculate intensity and other quantities.

    Returns
    -------
    lambda_line: float
            rest-frame wavelength of the line.
    """

    if line_name[0:2] == "CO":
        line_name_len = len(line_name)
        if line_name_len == 4:
            J_ladder = int(line_name[2])
        if line_name_len == 5 or line_name_len == 6:
            J_ladder = int(line_name[2:4])

        lambda_CO10 = 2610

        lambda_line = lambda_CO10 / J_ladder

    else:
        num = ""
        for i in line_name:
            if i.isdigit():
                num = num + i
        lambda_line = int(num)

    return lambda_line


def lambda_to_nu(lambda_line):
    """
    Converts wavelength to frequency

    parameters
    ----------
    lambda_line
               wavelength of lines in micro-meter.

    Returns
    -------
    frequency in GHz.

    """

    nu = c_in_m / lambda_line / 1e-6 / 1e9  # in GHz
    return nu


def nu_rest(line_name="CII158"):
    """
    Rest frame frequency of lines.
    """
    wavelength = lambda_line(line_name=line_name)
    nu = lambda_to_nu(wavelength)
    return nu


def nu_obs_to_z(nu_obs, line_name="CII158"):
    """
    This function evaluates the redshift of a particular line emission
    corresponding to the observed frequency.

    return: redshift of line emission.
    """

    global nu_rest_line

    nu_rest_line = nu_rest(line_name=line_name)

    if nu_obs >= nu_rest_line:
        z = 0

    else:
        z = (nu_rest_line / nu_obs) - 1
    return z



#Define a dictionary 
line_params_default = {
    'use_scatter': False,
    'a_off': a_off,
    'a_std': a_std,
    'b_off': b_off,
    'b_std': b_std,
    'f_duty': f_duty,
    'zg': zg,
    'scatter_dex':scatter_dex,
    'cov_scatter_matrix': cov_scatter_matrix
}
