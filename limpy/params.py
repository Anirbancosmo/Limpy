#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:55:51 2020
@author: anirbanroy
"""
import importlib

import camb
import numpy as np

import limpy.cosmos as cosmos
import limpy.inputs as inp

importlib.reload(inp)


# parameters

line_name = inp.line_params["line_name"]
Mmin = inp.astro_params["Mmin"]
Mmax = inp.astro_params["Mmax"]
Lsun = inp.astro_params["Lsun"]
delta_c = inp.astro_params["delta_c"]
Halo_model = inp.astro_params["halo_model"]
Halo_model_mdef = inp.astro_params["halo_mass_def"]
small_h = inp.cosmo_params["h"]
omega_matter = inp.cosmo_params["omega_mh2"] / small_h**2
omega_lambda = inp.cosmo_params["omega_lambda"]
use_scatter = inp.code_params["use_scatter"]
lcp_low = inp.default_dummy_values["lcp_low"]


# Initialize CAMB params
pars = camb.CAMBparams()

# Initialize cosmology for the LIMPY code
cosmo = cosmos.cosmo()

# Name of all lines
line_list = [
    "CI371",
    "CI610",
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
    "NII122",
    "NII205",
    "NIII57",
    "OI63",
    "OI145",
    "OIII52",
    "OIII88",
]


def model_avail(line_name="CII158", do_print=False):
    """
    Gives the available model names for a particularl line name.

    parameters
    ----------
    line_name: str
            name of the line to calculate intensity and other quantities.

    Returns
    -------
    sfr_model: str
            Available star formation models.

    model_name: str
            Available model names that converts the star formation rate to
            line luminosity or halo mass to line luminosity.
    """

    if line_name not in line_list:
        raise ValueError("Line name is not known. Check available lines.")

    sfr_models = ["Behroozi19", "Tng300", "Tng100", "Silva15", "Fonseca16"]

    if line_name == "CII158":
        model_names = [
            "Visbal10",
            "Silva15-m1",
            "Silva15-m2",
            "Silva15-m3",
            "Silva15-m4",
            "Padmanabhan18",
            "Fonseca16",
            "Lagache18",
            "Schaerer20",
            "Alma_scalling",
        ]

    if line_name == "CO10":
        model_names = ["Visbal10", "Kamenetzky15", "Padmanabhan18", "Alma_scalling"]

    if (
        line_name == "CO21"
        or line_name == "CO32"
        or line_name == "CO43"
        or line_name == "CO54"
        or line_name == "CO65"
        or line_name == "CO76"
        or line_name == "CO87"
        or line_name == "C98"
        or line_name == "CO109"
        or line_name == "1110"
        or line_name == "CO1211"
        or line_name == "CO1312"
    ):

        model_names = ["Visbal10", "Kamenetzky15", "Padmanabhan18", "Alma_scalling"]

    if line_name == "NII122":
        model_names = ["Visbal10"]

    if line_name == "NII205":
        model_names = ["Visbal10"]

    if line_name == "NIII57":
        model_names = ["Visbal10"]

    if line_name == "OI63":
        model_names = ["Visbal10"]

    if line_name == "OI145":
        model_names = ["Visbal10"]

    if line_name == "OIII52":
        model_names = ["Visbal10"]

    if line_name == "OIII88":
        model_names = [
            "Visbal10",
            "Delooze14",
            "Gong17",
            "Fonseca16",
            "Kannan22",
            "Alma_scalling",
        ]
    
    if do_print:
        print("The models available for %s lines\n" % (line_name))
        print("\n The star formation models are :", (sfr_models))
        print("\n The luminosity models are :", (model_names))

    return sfr_models, model_names


def line_scattered_params(line_name="CII158"):
    if line_name == "CII158":
        a_off = inp.default_L_CII_scatter_params["a_off"]
        a_std = inp.default_L_CII_scatter_params["a_std"]
        b_off = inp.default_L_CII_scatter_params["b_off"]
        b_std = inp.default_L_CII_scatter_params["b_std"]

    if line_name == "OIII88":
        a_off = inp.default_L_OIII_scatter_params["a_off"]
        a_std = inp.default_L_OIII_scatter_params["a_std"]
        b_off = inp.default_L_OIII_scatter_params["b_off"]
        b_std = inp.default_L_OIII_scatter_params["b_std"]

    if line_name == "CO10":
        a_off = inp.default_L_CO_1_0_scatter_params["a_off"]
        a_std = inp.default_L_CO_1_0_scatter_params["a_std"]
        b_off = inp.default_L_CO_1_0_scatter_params["b_off"]
        b_std = inp.default_L_CO_1_0_scatter_params["b_std"]
    if line_name == "CO21":
        a_off = inp.default_L_CO_2_1_scatter_params["a_off"]
        a_std = inp.default_L_CO_2_1_scatter_params["a_std"]
        b_off = inp.default_L_CO_2_1_scatter_params["b_off"]
        b_std = inp.default_L_CO_2_1_scatter_params["b_std"]
    if line_name == "CO32":
        a_off = inp.default_L_CO_3_2_scatter_params["a_off"]
        a_std = inp.default_L_CO_3_2_scatter_params["a_std"]
        b_off = inp.default_L_CO_3_2_scatter_params["b_off"]
        b_std = inp.default_L_CO_3_2_scatter_params["b_std"]
    if line_name == "CO43":
        a_off = inp.default_L_CO_4_3_scatter_params["a_off"]
        a_std = inp.default_L_CO_4_3_scatter_params["a_std"]
        b_off = inp.default_L_CO_4_3_scatter_params["b_off"]
        b_std = inp.default_L_CO_4_3_scatter_params["b_std"]
    if line_name == "CO54":
        a_off = inp.default_L_CO_5_4_scatter_params["a_off"]
        a_std = inp.default_L_CO_5_4_scatter_params["a_std"]
        b_off = inp.default_L_CO_5_4_scatter_params["b_off"]
        b_std = inp.default_L_CO_5_4_scatter_params["b_std"]
    if line_name == "CO65":
        a_off = inp.default_L_CO_6_5_scatter_params["a_off"]
        a_std = inp.default_L_CO_6_5_scatter_params["a_std"]
        b_off = inp.default_L_CO_6_5_scatter_params["b_off"]
        b_std = inp.default_L_CO_6_5_scatter_params["b_std"]
    if line_name == "CO76":
        a_off = inp.default_L_CO_7_6_scatter_params["a_off"]
        a_std = inp.default_L_CO_7_6_scatter_params["a_std"]
        b_off = inp.default_L_CO_7_6_scatter_params["b_off"]
        b_std = inp.default_L_CO_7_6_scatter_params["b_std"]
    if line_name == "CO87":
        a_off = inp.default_L_CO_8_7_scatter_params["a_off"]
        a_std = inp.default_L_CO_8_7_scatter_params["a_std"]
        b_off = inp.default_L_CO_8_7_scatter_params["b_off"]
        b_std = inp.default_L_CO_8_7_scatter_params["b_std"]
    if line_name == "CO98":
        a_off = inp.default_L_CO_9_8_scatter_params["a_off"]
        a_std = inp.default_L_CO_9_8_scatter_params["a_std"]
        b_off = inp.default_L_CO_9_8_scatter_params["b_off"]
        b_std = inp.default_L_CO_9_8_scatter_params["b_std"]
    if line_name == "CO109":
        a_off = inp.default_L_CO_10_9_scatter_params["a_off"]
        a_std = inp.default_L_CO_10_9_scatter_params["a_std"]
        b_off = inp.default_L_CO_10_9_scatter_params["b_off"]
        b_std = inp.default_L_CO_10_9_scatter_params["b_std"]
    if line_name == "CO1110":
        a_off = inp.default_L_CO_11_10_scatter_params["a_off"]
        a_std = inp.default_L_CO_11_10_scatter_params["a_std"]
        b_off = inp.default_L_CO_11_10_scatter_params["b_off"]
        b_std = inp.default_L_CO_11_10_scatter_params["b_std"]

    return a_off, a_std, b_off, b_std


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

def freq_to_lambda(nu):
    "frequency to wavelength converter."
    wav = c_in_mpc / nu
    return wav / small_h  # in mpc/h unit



cross_z1 = {
    "220": "CII158",
    "410": "OII88",
    "150": "CO1110",
    "90" : "CO76"}


cross_z2 = {
    "280": "CII158",
    "220": "CO1312",
    "150": "CO98",
    "90" : "CO54"}


cross_z3 = {
    "350": "CII158",
    "90" : "CO43",
    "150": "CO76",
    "220": "CO109",
    "280": "CO1312"}

cross_z4 = {
    "410": "CII158",
    "280": "CO1110",
    "220": "CO98",
    "150": "CO65"}


cross_z5 = {
    "90": "CO32",
    "150": "CO54",
    "280": "CO98",
    "350": "CO1211" }


cross_z6 = {
    "90": "CO21",
    "220": "CO54",
    "350": "CO87",
    "410": "CO98"}


cross_z7 ={
    "150": "CO32",
    "410": "CO87" }

cross_z8 = {
    "150": "CO21",
    "220": "CO32",
    "350": "CO54" }


cross_all = {
        'cross_z1': cross_z1, 
        'cross_z2': cross_z2,
        'cross_z3': cross_z3,
        'cross_z4': cross_z4,
        'cross_z5': cross_z5,
        'cross_z6': cross_z6,
        'cross_z7': cross_z7,
        'cross_z8': cross_z8,
        }


def cross_lines(line_group = None):

    if line_group == None:
        return cross_all
        
    else:
        return cross_all[line_group]

# Constants
c_in_m = 3e8  # meter/s
c_in_mpc = 9.72e-15  # Mpc/s
kpc_to_m = 3.086e19  # meter
mpc_to_m = 3.086e22  # meter
m_to_mpc = 3.24e-23  # Mpc
km_to_m = 1e3  # meter
km_to_mpc = 3.2408e-20
jy_unit = 1e-26  # Watts. m^{-2} HZ^{-1}
jy_bi_sr = 1.2566e-25  # Watts. m^{-2} HZ^{-1} or kg s^{-2}
Ghz_to_hz = 1e9  # Giga-Hertz to Hertz
k_b = 1.38e-23  # J/K
Lsun_erg_s = 3.828 * 1e33  # rg/s
minute_to_degree: 1.0 / 60
degree_to_minute: 60.0
degree_to_radian = np.pi / 180
nu_rest_CII = 1900  # GHZ
nu_rest_CO10 = 115  # GHZ
degree_to_minute = 60.0
minute_to_degree = 1.0 / 60
minute_to_radian = np.pi / 10800
degree_to_radian = np.pi / 180
