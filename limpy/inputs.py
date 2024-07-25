#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Limpy Parameters Module

This module defines default parameters, cosmological parameters, and a list of flags used to specify output options for the Limpy code.

This file is processed internally by the params.py file, which executes the Limpy code to generate specific outputs.
"""

# Cosmological Parameters
line_name = "CII158"
h = 0.6776
omega_lambda = 0.6889
omega_b = 0.0489
omega_m = 0.3111
omega_k = 0.0
tau = 0.056
ns = 0.9665
sigma_8 = 0.81

# Line Parameters
use_scatter = False
a_off = 6.98
a_std = 0.16
b_off = 0.99
b_std = 0.09
f_duty = 0.1
zg = 8.8
scatter_dex = 0.2
cov_scatter_matrix = None

# Astrophysical Parameters
M_min = 1e8
M_max = 1e19
delta_c = 1.68
halo_model = "sheth99"
halo_mass_def = "fof"
bias_model = "sheth01"
bias_mass_def = "vir"


# CIB parameters
# To implement  Eq(2) of https://arxiv.org/pdf/1907.11731.pdf
L_IR_cib = 10**11.6
Td_cib = 37
beta_cib= 1.5

# Lco average 
L_co = 1e5 



# Dummy Values to Extend the SFR-Mhalo Relation When Halomass Goes Below M_min
lcp_low = 1e-12

# Constants
c_in_m = 299792458.0  # meter
G_const = 6.67 * 1e-11  # kg^{-2} m^2
ghz_to_hz = 1e9  # Giga-Hertz to Hertz
Lsun = 3.828e26  # watts
mpc_to_m = 3.086e22  # meter
m_to_mpc = 3.24e-23  # Mpc
c_in_mpc = 9.72e-15  # Mpc/s
km_to_mpc = 3.2408e-20
km_to_m = 1e3  # meter
minute_to_degree = 1.0 / 60
degree_to_minute = 60.0
GHz_to_Hz = 1e9
Jy = 1e-26  # Watts. m^{-2} HZ^{-1}
kb = 1.380649e-23  # J/K
h_planck = 6.62607015e-34 #J/Hz
Lsun_erg_s = 3.828 * 1e33  # rg/s
minute_to_degree = 1.0 / 60
degree_to_minute = 60.0
nu_rest_CII = 1900  # GHZ
nu_rest_CO10 = 115  # GHZ
degree_to_minute = 60.0
minute_to_degree = 1.0 / 60
minute_to_radian = 0.000290888208
degree_to_radian = 0.0174533






###############################################################################
# Functions to dea with default input parameters
###############################################################################

def freq_to_lambda(nu):
    """
    Converts frequency to wavelength.

    Parameters
    ----------
    nu : float
        Frequency in GHz.

    Returns
    -------
    float
        Wavelength in mpc/h units.
    """
    wav = c_in_mpc / nu
    return wav

def lambda_line(line_name="CI371"):
    """
    Calculates the rest-frame wavelength of a line.

    Parameters
    ----------
    line_name : str
        Name of the line.

    Returns
    -------
    float
        Rest-frame wavelength of the line.
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
    Converts wavelength to frequency.

    Parameters
    ----------
    lambda_line : float
        Wavelength of lines in micrometers.

    Returns
    -------
    float
        Frequency in GHz.
    """
    nu = c_in_m / lambda_line / 1e-6 / 1e9  # in GHz
    return nu

def nu_rest(line_name="CII158"):
    """
    Calculates the rest frame frequency of a line.

    Parameters
    ----------
    line_name : str
        Name of the line.

    Returns
    -------
    float
        Rest frame frequency of the line.
    """
    wavelength = lambda_line(line_name=line_name)
    nu = lambda_to_nu(wavelength)
    return nu

def nu_obs_to_z(nu_obs, line_name="CII158"):
    """
    Evaluates the redshift of a particular line emission corresponding to the observed frequency.

    Parameters
    ----------
    nu_obs : float
        Observed frequency.
    line_name : str, optional
        Name of the line (default is "CII158").

    Returns
    -------
    float
        Redshift of line emission.
    """
    global nu_rest_line
    nu_rest_line = nu_rest(line_name=line_name)
    if nu_obs >= nu_rest_line:
        z = 0
    else:
        z = (nu_rest_line / nu_obs) - 1
    return z

parameters_default = {
    'use_scatter': False,
    'a_off': a_off,
    'a_std': a_std,
    'b_off': b_off,
    'b_std': b_std,
    'f_duty': f_duty,
    'zg': zg,
    'scatter_dex': scatter_dex,
    'cov_scatter_matrix': cov_scatter_matrix,
    "h": h,
    "omega_lambda": omega_lambda,
    "omega_b": omega_b,
    "omega_m": omega_m,
    "omega_k": omega_k,
    "tau": tau,
    "ns": ns,
    "sigma_8": sigma_8,
    "M_min": M_min,
    "M_max": M_max,
    "delta_c": delta_c,
    "halo_model": halo_model,
    "halo_mass_def": halo_mass_def,
    "bias_model": bias_model,
    "bias_mass_def": bias_mass_def,
    
    "L_IR_cib": L_IR_cib,
    "Td_cib" : Td_cib,
    "beta_cib" : beta_cib,
    
    "L_co": L_co
}




int_line_list = [
    "CII158",
    "CO10",
    "CO21",
    "CO32",
    "CO43",
    "CO54",
    "CO65",
    "CO76",
    "CO87",
    "CO98",
    "CO109",
    "CO1110",
    "CO1211",
    "CO1312",
    "OIII88",
]




color_list =[
"#892f70",
"#01d66e",
"#4c7cff",
"#d54708",
"#69d6dc",
"#953112",
"#b7b5ff",
"#7a5900",
"#ff669e",
"#ff8e88"]


# Scaling co-efficients for CII 158 lines
a_off_CII_158 = 6.98
a_std_CII_158 = 0.16
b_off_CII_158 = 0.99
b_std_CII_158 = 0.09

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

# For OIII
a_off_OIII_88 = 7.4
a_std_OIII_88 = 0.16
b_off_OIII_88 = 0.97
b_std_OIII_88 = 0.09

