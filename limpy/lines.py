#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division

import matplotlib as pl
import numpy as np
import os
import os.path
from astropy.convolution import Gaussian2DKernel, convolve

#from astropy.modeling.models import Gaussian2D
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import simpson as simps
import limpy.utils as lu
import limpy.inputs as inp
from scipy.special import erf

pl.rcParams["xtick.labelsize"] = "10"
pl.rcParams["ytick.labelsize"] = "10"
pl.rcParams["axes.labelsize"] = "15"
pl.rcParams["axes.labelsize"] = "15"


import limpy.cosmos as cosmos

small_h = cosmos.cosmo().h

################################################################################
##     Interpolate the sfr for different models at the beginning
################################################################################


################################################################################
##      SFR and line luminosity models
################################################################################


# File paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
sfr_file_paths = {
    "Behroozi19": os.path.join(DATA_DIR, 'sfr_Behroozi.dat'),
    "Tng100": os.path.join(DATA_DIR, 'sfr_processed_TNG100-1.npz'),
    "Tng300": os.path.join(DATA_DIR, 'sfr_processed_TNG300-1.npz')
}



def get_sfr_interpolation(sfr_model):
    """
    Load SFR data and create interpolation function if not already cached.

    Parameters:
    - sfr_model (str): The SFR model to use. Available options are "Behroozi19", "Tng100", or "Tng300".

    Returns:
    - RectBivariateSpline: The interpolation function for the specified SFR model.
    """
    # Dictionary to hold interpolation functions for each model
    sfr_interpolations = {}
    
    if sfr_model in sfr_interpolations:
        return sfr_interpolations[sfr_model]
    
    if sfr_model == "Behroozi19":
        z, m, sfr_data = np.loadtxt(sfr_file_paths[sfr_model], unpack=True)
        zlen = 137  # manually checked
        mlen = int(len(z) / zlen)
        zn = z[:zlen]
        log_mh = np.log10(m.reshape(mlen, zlen)[:, 0])
        sfr_int = sfr_data.reshape(mlen, zlen)
        sfr_interpolations[sfr_model] = RectBivariateSpline(log_mh, zn, sfr_int)
    else:
        f = np.load(sfr_file_paths[sfr_model])
        sfr_interpolations[sfr_model] = RectBivariateSpline(f["halomass"], f["z"], f["sfr"])
        f.close()
    
    return sfr_interpolations[sfr_model]

def mhalo_to_sfr(Mhalo, z, sfr_model = "Behroozi19"):
    """
    Returns the SFR history for discrete values of halo mass.

    Parameters:
    - Mhalo (numpy.ndarray): Array of halo masses.
    - z (numpy.ndarray): Array of redshifts.
    - sfr_model (str): The SFR model to use. Default is "Behroozi19". 
      Available options are "Behroozi19", "Tng100", or "Tng300".

    Returns:
    - numpy.ndarray: Array of SFR values corresponding to the input halo masses and redshifts.
    """
    
    
    
    if sfr_model not in ["Behroozi19", "Tng100", "Tng300", "Silva15", "Fonseca16"]:
        raise ValueError("Input SFR model is not implemented in the package. \
                         Please either implement the SFR model in the code  \
                         or check the spelling of the input SFR model.")
    
    
    if sfr_model == "Silva15":
        sfr = sfr_silva15(Mhalo, z)
    elif sfr_model == "Fonseca16":
        sfr = sfr_Fonseca16(Mhalo, z)
    else:
        sfr_interpolation = get_sfr_interpolation(sfr_model)
        sfr = sfr_interpolation(np.log10(Mhalo), z)
    
    res = np.where(sfr < 1e-4, inp.lcp_low, sfr)
    return res.flatten()



def sfr_Fonseca16(M, z):
    """
    Star formation rate model of Fonseca16.
    Arxiv: https://arxiv.org/abs/1607.05288

    parameters
    ----------
    M
      Mass of halos in Msun/h
    z
     Redshift of halos.

    Returns
    -------
    star formation rate in Msun/year

    """
    M= M / small_h #Msun


    Ma = 1e8
    zs_fit = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    M0_fit = [
        3.0e-10,
        1.7e-09,
        4.0e-09,
        1.1e-08,
        6.6e-08,
        7.0e-07,
        7.0e-07,
        7.0e-07,
        7.0e-07,
        7.0e-07,
        7.0e-07,
    ]
    Mb_fit = [
        6.0e10,
        9.0e10,
        7.0e10,
        5.0e10,
        5.0e10,
        6.0e10,
        6.0e10,
        6.0e10,
        6.0e10,
        6.0e10,
        6.0e10,
    ]
    Mc_fit = [
        1.0e12,
        2.0e12,
        2.0e12,
        3.0e12,
        2.0e12,
        2.0e12,
        2.0e12,
        2.0e12,
        2.0e12,
        2.0e12,
        2.0e12,
    ]
    a_fit = [3.15, 2.9, 3.1, 3.1, 2.9, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
    b_fit = [-1.7, -1.4, -2.0, -2.1, -2.0, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6]
    c_fit = [-1.7, -2.1, -1.5, -1.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

    M0 = np.interp(z, zs_fit, np.log10(M0_fit))
    Mb = np.interp(z, zs_fit, np.log10(Mb_fit))
    Mc = np.interp(z, zs_fit, np.log10(Mc_fit))
    a = np.interp(z, zs_fit, a_fit)
    b = np.interp(z, zs_fit, b_fit)
    c = np.interp(z, zs_fit, c_fit)

    M0 = 10**M0
    Mb = 10**Mb
    Mc = 10**Mc

    return M0 * (M / Ma) ** a * (1 + M / Mb) ** b * (1 + M / Mc) ** c


def sfr_silva15(M, z):
    """
    Star formation rate model of Silva15
    Arxiv: https://arxiv.org/abs/1410.4808

    parameters
    ----------
    M
      Mass of halos in Msun/h
    z
     Redshift of halos.

    Returns
    -------
    star formation rate in Msun/year

    """

    M= M / small_h #Msun

    zs_fit = [
        0.0,
        0.25,
        0.5,
        0.51,
        1.07,
        1.63,
        2.19,
        2.75,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.25,
        10.0,
        12.0,
        13.0,
        14.75,
        16.5,
        18.25,
        20.0,
    ]

    M0_fit = [
        7.99834255e-10,
        1.05681751e-09,
        1.39636836e-09,
        1.41201724e-09,
        2.63560304e-09,
        4.91948907e-09,
        9.18248019e-09,
        1.71395731e-08,
        3.30000000e-08,
        1.70000000e-07,
        9.00000000e-07,
        3.60000000e-06,
        6.60000000e-06,
        1.00000000e-05,
        3.70000000e-05,
        5.00000000e-05,
        5.00000000e-05,
        6.75000000e-05,
        8.50000000e-05,
        1.02500000e-04,
        1.20000000e-04,
    ]

    Ma_fit = 1e8 * np.ones(len(zs_fit))

    Mb_fit = [
        8.00000e11,
        8.00000e11,
        8.00000e11,
        8.00000e11,
        8.00000e11,
        8.00000e11,
        8.00000e11,
        8.00000e11,
        4.00000e11,
        3.00000e11,
        3.00000e11,
        2.00000e11,
        1.60000e11,
        1.70000e11,
        1.70000e11,
        1.50000e11,
        1.50000e11,
        1.52625e11,
        1.55250e11,
        1.57875e11,
        1.60500e11,
    ]

    a_fit = [
        2.7,
        2.7,
        2.7,
        2.7,
        2.7,
        2.7,
        2.7,
        2.7,
        2.7,
        2.6,
        2.4,
        2.25,
        2.25,
        2.25,
        2.1,
        2.1,
        2.1,
        2.1,
        2.1,
        2.1,
        2.1,
    ]

    b_fit = [
        -4.0,
        -4.0,
        -4.0,
        -4.0,
        -4.0,
        -4.0,
        -4.0,
        -4.0,
        -3.4,
        -3.1,
        -2.3,
        -2.3,
        -2.3,
        -2.4,
        -2.2,
        -2.2,
        -2.2,
        -2.2525,
        -2.305,
        -2.3575,
        -2.41,
    ]

    M0 = np.interp(z, zs_fit, np.log10(M0_fit))
    Ma = np.interp(z, zs_fit, np.log10(Ma_fit))
    Mb = np.interp(z, zs_fit, np.log10(Mb_fit))
    a = np.interp(z, zs_fit, a_fit)
    b = np.interp(z, zs_fit, b_fit)

    return 10**M0 * (M / 10**Ma) ** a * (1 + M / 10**Mb) ** b


def LCII_Silva15(M, z, sfr_model="Behroozi19", model_name="Silva15-m1"):
    """
    CII line luminosity model of Silva15
    Arxiv: https://arxiv.org/abs/1410.4808

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    sfr_model: string
        star formation model name.

    model_name
        model name to convert sfr to line luminosity.

    Returns
    -------
    Luminosity of CII158 line in Lsun unit.

    """
    M= M / small_h #Msun

    sfr = mhalo_to_sfr(M, z, sfr_model=sfr_model)

    if model_name == "Silva15-m1":
        mp = [0.8475, 7.2203]

    if model_name == "Silva15-m2":
        mp = [1.0000, 6.9647]

    if model_name == "Silva15-m3":
        mp = [0.8727, 6.7250]

    if model_name == "Silva15-m4":
        mp = [0.9231, 6.5234]

    return 10 ** (mp[0] * np.log10(sfr) + mp[1])


def LCII_Fonseca16(M, z, sfr_model="Behroozi19"):
    """
    CII158 luminosity model of Fonseca16.
    Arxiv: https://arxiv.org/abs/1410.4808

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    sfr_model: string
        star formation model name.


    Returns
    -------
    Luminosity of CII158 line in Lsun unit.

    """
    M= M / small_h #Msun

    sfr = mhalo_to_sfr(M, z, sfr_model=sfr_model)
    return 1.322e7 * (sfr) ** 1.02


def L_CII158_Visbal10(M, z, f_duty=0.1, sfr_model=None):
    """
    CII158 line luminosity model of Visbal10.
    Arxiv: https://arxiv.org/abs/1008.3178v2

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    sfr_model: string
        star formation model name.

    Returns
    -------
    Luminosity of CII158 line in Lsun unit.

    """

    M= M / small_h #Msun
    R = 6e6
    
    if sfr_model != "Visbal10":
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out * f_duty

    elif sfr_model == "Visbal10":
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * f_duty
    return L


def L_CO_Visbal10(M, z, f_duty=0.1, J_ladder=1, sfr_model="Visbal10"):
    """
    CO j-transitions line luminosity model of Visbal10.
    Arxiv: https://arxiv.org/abs/1008.3178v2

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    f_futy
        duty cycle. Default value is set to 0.1

    Returns
    -------
    Luminosity of CII158 line in Lsun unit.

    """
    M= M / small_h #Msun

    R_array = [
        3.7e3,
        2.8e4,
        7e4,
        9.7e4,
        9.6e4,
        9.5e4,
        8.9e4,
        7.7e4,
        6.9e4,
        5.3e4,
        3.8e4,
        2.6e4,
        1.4e4,
    ]

    R = R_array[J_ladder-1]
    #print("J ladder is", R)
        
    
    if sfr_model != "Visbal10":
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out
        
    elif sfr_model == "Visbal10":
    	L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * f_duty

    return L


def L_NII205_Visbal10(M, z, f_duty=0.1, sfr_model=None):
    """
    NII205 line luminosity model of Visbal10.
    Arxiv: https://arxiv.org/abs/1008.3178v2

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    f_futy
        duty cycle. Default value is set to 0.1 .


    Returns
    -------
    Luminosity of NII205 line in Lsun unit.

    """

    M= M / small_h #Msun

    R = 2.5e5

    if sfr_model != "Visbal10":
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    elif sfr_model == "Visbal10":
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 *  f_duty

    return L


def L_NII122_Visbal10(M, z, f_duty=0.1, sfr_model=None):
    """
    NII122 line luminosity model of Visbal10.
    Arxiv: https://arxiv.org/abs/1008.3178v2

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    f_futy
        duty cycle. Default value is set to 0.1 .


    Returns
    -------
    Luminosity of NII122 line in Lsun unit.

    """

    M= M / small_h #Msun

    R = 7.9e5

    if sfr_model != "Visbal10":
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    elif sfr_model == "Visbal10":
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 *  f_duty
    return L


def L_CI610_Visbal10(M, z, f_duty=0.1, sfr_model=None):
    """
    CI610 line luminosity model of Visbal10.
    Arxiv: https://arxiv.org/abs/1008.3178v2

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    f_futy
        duty cycle. Default value is set to 0.1 .


    Returns
    -------
    Luminosity of CI610 line in Lsun unit.

    """
    M= M / small_h #Msun

    R = 1.4e4

    if sfr_model != "Visbal10":
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    elif sfr_model == "Visbal10":
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 *  f_duty
    return L


def L_CI371_Visbal10(M, z, f_duty=0.1, sfr_model=None):
    """
    CI371 line luminosity model of Visbal10.
    Arxiv: https://arxiv.org/abs/1008.3178v2

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    sfr_model: string
        star formation model name.


    Returns
    -------
    Luminosity of CI371 line in Lsun unit.

    """

    M= M / small_h #Msun

    R = 4.8e4

    if sfr_model != "Visbal10":
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    elif sfr_model == "Visbal10":
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 *  f_duty
    return L


def L_OIII88_Visbal10(M, z, f_duty=0.1, sfr_model=None):
    """
    OIII88 line luminosity model of Visbal10.
    Arxiv: https://arxiv.org/abs/1008.3178v2

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    sfr_model: string
        star formation model name.


    Returns
    -------
    Luminosity of OIII88 line in Lsun unit.

    """

    M= M / small_h #Msun

    R = 2.3e6

    if sfr_model != "Visbal10":
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    elif sfr_model == "Visbal10":
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * f_duty
    return L


def L_OIII88_Harikane20(M, z, sfr_model="Behroozi19"):
    """
    OIII88 line luminosity model of Harikane20.
    Arxiv: https://arxiv.org/abs/1910.10927

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    sfr_model: string
        star formation model name.


    Returns
    -------
    Luminosity of OIII88 line in Lsun unit.

    """
    M= M / small_h #Msun

    sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
    L = 0.97 * np.log10(sfr_out) + 7.4
    return 10**L


def L_OIII88_Gong17(M, z, sfr_model="Behroozi19"):
    """
    OIII88 line luminosity model of Gong17.
    Arxiv: https://arxiv.org/abs/1610.09060

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    sfr_model: string
        star formation model name.

    Returns
    -------
    Luminosity of OIII88 line in Lsun unit.

    """

    """
    https://arxiv.org/abs/1610.09060, Eq 4
    """

    M= M / small_h #Msun

    sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
    factor = 7.6e-42
    L = sfr_out / factor / inp.Lsun_erg_s
    return L


def L_Oxygen_Fonseca16(M, z, line_name="OIII88", sfr_model="Behroozi19"):
    """
    Oxygen lines luminosity model of Gong17.
    Arxiv: https://arxiv.org/pdf/1607.05288.pdf, Eq (8) and (24)

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    line_name: str
        line names such as OIII88, OIII52, OI145, OI63.

    sfr_model: string
        star formation model name.

    Returns
    -------
    Luminosity of OIII88 line in Lsun unit.

    """

    M= M / small_h #Msun

    sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)

    if line_name == "OIII88":
        factor, gamma = 10**40.44, 0.98

    if line_name == "OIII52":
        factor, gamma = 10**40.52, 0.88

    if line_name == "OI145":
        factor, gamma = 10**39.54, 0.89

    if line_name == "OI63":
        factor, gamma = 10**40.60, 0.98

    L = (factor / inp.Lsun_erg_s) * (sfr_out) ** gamma
    return L


def L_OIII52_Visbal10(M, z, f_duty=0.1, sfr_model=None):
    """
    OIII52 line luminosity model of Visbal10.
    Arxiv: https://arxiv.org/abs/1008.3178v2

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    f_futy
        duty cycle. Default value is set to 0.1 .


    Returns
    -------
    Luminosity of OIII52 line in Lsun unit.

    """

    M= M / small_h #Msun

    fstar = 0.1
    R = 3e6

    if sfr_model != None:
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    else:
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * (fstar / f_duty)

    return L


def L_OI145_Visbal10(M, z, f_duty=0.1, sfr_model=None):
    """
    OI145 line luminosity model of Visbal10.
    Arxiv: https://arxiv.org/abs/1008.3178v2

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    f_futy
        duty cycle. Default value is set to 0.1 .


    Returns
    -------
    Luminosity of OI145 line in Lsun unit.

    """

    M= M / small_h #Msun

    fstar = 0.1
    R = 3.3e5

    if sfr_model != None:
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    else:
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * (fstar / f_duty)

    return L


def L_OI63_Visbal10(M, z, f_duty=0.1, sfr_model=None):
    """
    OI63 line luminosity model of Visbal10.
    Arxiv: https://arxiv.org/abs/1008.3178v2

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    f_futy
        duty cycle. Default value is set to 0.1 .


    Returns
    -------
    Luminosity of OI63 line in Lsun unit.

    """

    M= M / small_h #Msun

    fstar = 0.1
    R = 3.8e5

    if sfr_model != None:
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out
    else:
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * (fstar / f_duty)
    return L


def L_NIII57_Visbal10(M, z, f_duty=0.1, sfr_model=None):
    """
    NIII57 line luminosity model of Visbal10.
    Arxiv: https://arxiv.org/abs/1008.3178v2

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    f_futy
        duty cycle. Default value is set to 0.1 .


    Returns
    -------
    Luminosity of NIII57 line in Lsun unit.

    """


    fstar = 0.1
    R = 2.4e6

    if sfr_model != None:
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out
    else:
        M= M / small_h #Msun

        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * (fstar / f_duty)

    return L


def L_lines_Thesan(M, z, line_name="OIII88", sfr_model="Behroozi19"):
    """
    Oxygen and CII lines luminosity model of Thesan project.
    Arxiv:  https://arxiv.org/pdf/2111.02411.pdf

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    line_name: str
        line names

    sfr_model: string
        star formation model name.

    Returns
    -------
    Luminosity of OIII88 line in Lsun unit.

    """

    sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)


    x = np.log10(sfr_out)
    xb = 0

    if line_name == "OII":
        a = 7.08
        ma = 1.11
        mb = 1.31
        mc = 0.64
        xc = 0.54

    # Probably not right. In this case, I am getting
    # very low luminsoity at z =6 for M = 10**11.
    if line_name == "CII158":
        a = 5.01
        ma = 1.49
        mb = 0.0
        mc = 0.0
        xc = 0.0

    if line_name == "NII122":
        a = 4.42
        ma = 1.83
        mb = 0.0
        mc = 0.0
        xc = 0.0

    if line_name == "OIII88":
        a = 6.98
        ma = 1.24
        mb = 1.19
        mc = 0.53
        xc = 0.66

    if line_name == "OIII52":
        a = 6.75
        ma = 1.47
        mb = 0
        mc = 0
        xc = 0

    if line_name == "OIII":
        a = 7.84
        ma = 1.24
        mb = 1.19
        mc = 0.53
        xc = 0.66

    L1 = a + ma * x
    L2 = a + (ma - mb) * xb + mb * x
    L3 = a + (ma - mb) * xb + (mb - mc) * xc + mc * x

    if xc == 0.0:
        res1 = np.where(x < xb, L1, x)
        L = np.where(x >= xb, L3, res1)

    else:
        res1 = np.where(x < xb, L1, x)
        res2 = np.where(((x >= xb) & (x < xc)), L2, res1)
        L = np.where(x >= xc, L3, res2)

    return 10**L


def L_line_Visbal10(M, z, f_duty=0.1, line_name="CO10", sfr_model= "Visbal10"):
    """
    Return the luminosity for many different lines based on Visbal10 model.
    Arxiv:  https://arxiv.org/pdf/2111.02411.pdf

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.

    line_name: str
        line names

    sfr_model: string
        star formation model name.
    f_duty:
        default value of f_duty = 0.1.

    Returns
    -------
    Luminosity of line in Lsun unit.

    """

    if line_name in lu.line_list:
        pass
    else:
        print(
            "line name is not found! Choose a line name from the list:\n", lu.line_list
        )

    if line_name[0:2] == "CO":
        line_name_len = len(line_name)
        if line_name_len == 4:
            J_ladder = int(line_name[2])
        elif line_name_len == 5 or line_name_len == 6:
            J_ladder = int(line_name[2:4])

        L_line = L_CO_Visbal10(
            M, z, f_duty=f_duty, sfr_model=sfr_model, J_ladder=J_ladder,
        )

    if line_name == "CII158":
        L_line = L_CII158_Visbal10(M, z, f_duty=f_duty, sfr_model=sfr_model)

    if line_name == "CI371":
        L_line = L_CI371_Visbal10(M, z, f_duty=f_duty, sfr_model=sfr_model)

    if line_name == "CI610":
        L_line = L_CI610_Visbal10(M, z, f_duty=f_duty, sfr_model=sfr_model)

    if line_name == "OI63":
        L_line = L_OI63_Visbal10(M, z, f_duty=f_duty, sfr_model=sfr_model)

    if line_name == "OI145":
        L_line = L_OI145_Visbal10(M, z, f_duty=f_duty, sfr_model=sfr_model)

    if line_name == "OIII52":
        L_line = L_OIII52_Visbal10(M, z, f_duty=f_duty, sfr_model=sfr_model)

    if line_name == "OIII88":
        L_line = L_OIII88_Visbal10(M, z, f_duty=f_duty, sfr_model=sfr_model)

    if line_name == "NII122":
        L_line = L_NII205_Visbal10(M, z, f_duty=f_duty, sfr_model=sfr_model)

    if line_name == "NIII57":
        L_line = L_NII205_Visbal10(M, z, f_duty=f_duty, sfr_model=sfr_model)

    return L_line




def LCO10_prime_padmanabhan18(Mhalo, z):
    """
    Luminosity of CO10 based on padmanabhan18 model.

    parameters
    ----------
    M
      Mass of halos in Msun/h

    z
     Redshift of halos.


    Returns
    -------
    Luminosity of CO10 line in Lsun unit.

    """

    Mhalo /= small_h
    Mhalo = np.array(Mhalo)
    # assert z<=3, "LCO-Mhalo relation is valid for redshift between 0 and 3."
    M10 = 4.17e12
    M11 = -1.17
    N10 = 0.0033
    N11 = 0.04
    b10 = 0.95
    b11 = 0.48
    y10 = 0.66
    y11 = -0.33
    M1_z = 10 ** (np.log10(M10) + (M11 * z) / (z + 1))
    N_z = N10 + (N11 * z) / (z + 1)
    b_z = b10 + (b11 * z) / (z + 1)
    y_z = y10 + (y11 * z) / (z + 1)

    Lco_prime = (2 * N_z * Mhalo) / ((Mhalo / M1_z) ** -b_z + (Mhalo / M1_z) ** y_z)

    return Lco_prime


def LCII_Lagache18(Mhalo, z, sfr_model="Behroozi19"):
    """
    Luminosity of CII158 based on Lagache18 model.

    parameters
    ----------
    M
        Mass of halos in Msun/h

    z
        Redshift of halos.

    sfr_model
        star formation rate model.

    Returns
    -------
    Luminosity of CII158 line in Lsun unit.

    """

    sfr_out = mhalo_to_sfr(Mhalo, z, sfr_model=sfr_model)
    factor = (1.4 - 0.07 * z) * np.log10(sfr_out) + 7.1 - 0.07 * z
    return 10**factor


def LCII_Lagache18_metal(Mhalo, z, zg=8.8, sfr_model="Behroozi19"):
    """
    Luminosity of CII158 based on padmanabhan18 model.

    parameters
    ----------
    M
        Mass of halos in Msun/h

    z
        Redshift of halos.

    sfr_model
        star formation rate model.

    Returns
    -------
    Luminosity of CII158 line in Lsun unit.

    """

    sfr_out = mhalo_to_sfr(Mhalo, z, sfr_model=sfr_model)
    factor = 7.0 + 1.2 * np.log10(sfr_out) + 0.021 * np.log10(zg) + 0.012 * np.log10(sfr_out) * np.log10(zg) - 0.74*(np.log10(zg))**2
    
    return 10**factor





def LCII_Schaerer20(Mhalo, z, sfr_model="Behroozi19"):
    """
    Luminosity of CII158 based on Schaerer20 model.

    parameters
    ----------
    M
        Mass of halos in Msun/h

    z
        Redshift of halos.

    sfr_model
        star formation rate model.

    Returns
    -------
    Luminosity of CII158 line in Lsun unit.

    """

    sfr_out = mhalo_to_sfr(Mhalo, z, sfr_model=sfr_model)
    factor = 6.43 + 1.26 * np.log10(sfr_out)
    return 10**factor


def LCII_padmanabhan18(Mhalo, z):
    """
    Luminosity of CII158 based on padmanabhan18 model.

    parameters
    ----------
    M
        Mass of halos in Msun/h

    z
        Redshift of halos.

    sfr_model
        star formation rate model.

    Returns
    -------
    Luminosity of CII158 line in Lsun unit.

    """

    Mhalo = np.array(Mhalo)
    M1 = 2.39e-5
    N1 = 4.19e11
    alpha = 1.79
    beta = 0.49
    F_z = ((1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)) ** alpha
    Lcii = F_z * (Mhalo / M1) ** beta * np.exp(-N1 / Mhalo)
    return Lcii


def L_FIR(Mhalo, z, FIR_model_name="Carilli11", sfr_model="Behroozi19"):
    sfr_out = mhalo_to_sfr(Mhalo, z, sfr_model=sfr_model)

    if FIR_model_name == "Carilli11":
        Lfir = 1.1 * 1e10 * sfr_out

    return Lfir


################################################################################
##      Luminosity for CO
################################################################################


def LCO_Kamenetzky15(
    Mhalo, z, line_name="CO10", sfr_model="Behroozi19", FIR_model="Carilli11"
):
    """
    Luminosity of CO10 based on Kamenetzky15 model.

    parameters
    ----------
    M
        Mass of halos in Msun/h

    z
        Redshift of halos.

    sfr_model
        star formation rate model.

    FIR_mode
        we fix it to Carilli11 model.

    Returns
    -------
    Luminosity of CO lines in Lsun unit.

    """

    LFIR = L_FIR(Mhalo, z, FIR_model_name=FIR_model, sfr_model=sfr_model)

    if line_name == "CO10":
        a, b = 1.27, -1
    elif line_name == "CO21":
        a, b = 1.11, 0.6
    elif line_name == "CO32":
        a, b = 1.18, 0.1
    elif line_name == "CO43":
        a, b = 1.09, 1.2
    elif line_name == "CO54":
        a, b = 1.05, 1.8
    elif line_name == "CO65":
        a, b = 1.04, 2.2
    elif line_name == "CO76":
        a, b = 0.98, 2.9
    elif line_name == "CO87":
        a, b = 1.00, 3.0
    elif line_name == "CO98":
        a, b = 1.03, 2.9
    elif line_name == "CO109":
        a, b = 1.01, 3.2
    elif line_name == "CO1110":
        a, b = 1.06, 3.1
    elif line_name == "CO1211":
        a, b = 0.99, 3.7
    elif line_name == "CO1312":
        a, b = 1.12, 2.9
    else:
        raise ValueError("Choose a Jup")
    Lco = 10 ** ((np.log10(LFIR) - b) / a)

    if line_name[0:2] == "CO":
        line_name_len = len(line_name)
    if line_name_len == 4:
        J_ladder = int(line_name[2])
    elif line_name_len == 5 or line_name_len == 6:
        J_ladder = int(line_name[2:4])

    return (4.9e-5) * Lco * J_ladder**3


def LCO_Padmanabhan18(Mhalo, z, line_name="CO10"):
    line_name_len = len(line_name)
    if line_name_len == 4:
        J_lader = int(line_name[2])
    elif line_name_len == 5 or line_name_len == 6:
        J_lader = int(line_name[2:4])

    Lco_prime = LCO10_prime_padmanabhan18(Mhalo, z)
    L_line = 4.9e-5 * (J_lader) ** 3 * Lco_prime  # Equation 4 of  arxiv:1706.03005

    return L_line


def Lline_delooze14(Mhalo, z, line_name="CII158", sfr_model="Behroozi19"):
    """
    Luminosity of CII158 line based on delooze14 model.

    parameters
    ----------
    M
        Mass of halos in Msun/h

    z
        Redshift of halos.

    sfr_model
        star formation rate model.

    FIR_mode
        we fix it to Carilli11 model.

    Returns
    -------
    Luminosity of CO lines in Lsun unit.

    """

    sfr_out = mhalo_to_sfr(Mhalo, z, sfr_model=sfr_model)

    if line_name == "CII158":
        alpha, beta = 0.93, -6.99

    if line_name == "OI63":
        alpha, beta = 1.41, -9.19

    if line_name == "OIII88":
        alpha, beta = 1.01, -7.33

    L_line = 10 ** ((np.log10(sfr_out) - beta) / alpha)
    return L_line


################################################################################


def sfr_to_L_line_alma(z, sfr, line_name="CII158", line_params=None):
    """
    Calculates luminosity of the OIII lines from SFR assuming a 3\sigma Gaussian
    scatter. The parameter values for the scattered relation
    are mentioned in default_params module.

    Input: z and sfr

    Return: luminosity of OIII lines in log scale
    """

    # Check if the line name is valid
    if line_name not in lu.line_list:
        raise ValueError("Not a familiar line.")

    # Get default parameters if line_params is not provided
    if line_params is None:
        line_params = lu.line_scattered_params_alma(line_name)
        a_off = line_params['a_off']
        b_off = line_params['b_off']
    
    
    else:
        # Merge line_params with default parameters
        defaults = lu.line_scattered_params_alma(line_name)
        line_params = {**defaults, **line_params}
        a_off = line_params['a_off']
        b_off = line_params['b_off']

    # Function to calculate L_CO_log
    def L_co_log(sfr, alpha, beta):
        nu_co_line = inp.nu_rest(line_name)
        L_ir_sun = sfr * 1e10
        L_coprime = (L_ir_sun * 10 ** (-beta)) ** (1 / alpha)
        L_co = 4.9e-5 * (nu_co_line / 115.27) ** 3 * L_coprime
        return np.log10(L_co)

    # Convert scalar inputs to arrays with at least one dimension
    sfr = np.atleast_1d(sfr)

    # Calculate log_L_line
    if line_name == "CII158" or line_name == "OIII88":
        log_L_line = a_off + b_off * np.log10(sfr)
        
    elif line_name.startswith("CO"):
        log_L_line = L_co_log(sfr, a_off, b_off)

    return 10 ** log_L_line



def mhalo_to_lline_alma(
    Mh,
    z,
    line_name="CII158",
    sfr_model="Behroozi19",
    line_params=None
):
    """
    This function returns luminosity of lines (following the input line_name) in the unit of L_sun.
    Kind optitions takes the SFR: mean , up (1-sigma upper bound) and
    down (1-sigma lower bound)
    """
    # if

    global sfr_cal, L_line

    sfr_cal = mhalo_to_sfr(Mh, z, sfr_model=sfr_model)

    L_line = sfr_to_L_line_alma(
        z,
        sfr_cal,
        line_name=line_name,
        line_params=line_params
    )

    return L_line


def mhalo_to_lcp_fit(
    Mhalo,
    z,
    model_name="Silva15-m1",
    sfr_model="Behroozi19",
    line_params=inp.line_params_default
):
    if (
        model_name == "Silva15-m1"
        or model_name == "Silva15-m2"
        or model_name == "Silva15-m3"
        or model_name == "Silva15-m4"
    ):
        LCII = LCII_Silva15(Mhalo, z, model_name=model_name, sfr_model=sfr_model)

    if model_name == "Padmanabhan18":
        LCII = LCII_padmanabhan18(Mhalo, z)

    if model_name == "Visbal10":
        LCII = L_CII158_Visbal10(Mhalo, z, sfr_model=sfr_model)

    if model_name == "Fonseca16":
        LCII = LCII_Fonseca16(Mhalo, z, sfr_model=sfr_model)

    if model_name == "Lagache18":
        LCII = LCII_Lagache18(Mhalo, z, sfr_model=sfr_model)
        
    if model_name == "Lagache18-metal":
        LCII = LCII_Lagache18_metal(Mhalo, z, zg = line_params['zg'], sfr_model=sfr_model)

    if model_name == "Schaerer20":
        LCII = LCII_Schaerer20(Mhalo, z, sfr_model=sfr_model)

    if model_name == "Alma_scalling":
        LCII = mhalo_to_lline_alma(
            Mhalo,
            z,
            line_name="CII158",
            sfr_model=sfr_model,
            line_params=line_params
        )

    return LCII


def mhalo_to_lco_fit(
    Mhalo,
    z,
    line_name="CO10",
    f_duty=0.1,
    model_name="Visbal10",
    sfr_model="Behroozi19",
    use_scatter=False,
    params_fisher=None,
):

    if model_name == "Visbal10":
        L_line = L_line_Visbal10(Mhalo, z, f_duty=f_duty, line_name=line_name, sfr_model=sfr_model)

    if model_name == "Padmanabhan18":
        L_line = LCO_Padmanabhan18(Mhalo, z, line_name=line_name)

    if model_name == "Kamenetzky15":
        L_line = LCO_Kamenetzky15(Mhalo, z, line_name=line_name, sfr_model=sfr_model)

    if model_name == "Alma_scalling":
        L_line = mhalo_to_lline_alma(
            Mhalo,
            z,
            line_name=line_name,
            sfr_model=sfr_model,
            use_scatter=use_scatter,
            params_fisher=params_fisher,
        )
    return L_line


def mhalo_to_Oxygen_fit(
    Mhalo,
    z,
    line_name="OIII88",
    f_duty=0.1,
    model_name="Visbal10",
    sfr_model="Behroozi19",
    use_scatter=False,
    params_fisher=None,
):

    if model_name == "Visbal10":
        L_line = L_line_Visbal10(Mhalo, z, f_duty=f_duty, line_name=line_name)

    if model_name == "Delooze14":
        L_line = Lline_delooze14(Mhalo, z, line_name=line_name, sfr_model=sfr_model)

    if model_name == "Harikane20":
        L_line = L_OIII88_Harikane20(Mhalo, z, sfr_model=sfr_model)

    if model_name == "Gong17":
        L_line = L_OIII88_Gong17(Mhalo, z, sfr_model=sfr_model)

    if model_name == "Fonseca16":
        L_line = L_Oxygen_Fonseca16(Mhalo, z, line_name=line_name, sfr_model=sfr_model)

    if model_name == "Kannan22":
        L_line = L_lines_Thesan(Mhalo, z, line_name=line_name, sfr_model=sfr_model)

    if model_name == "Alma_scalling":
        L_line = mhalo_to_lline_alma(
            Mhalo,
            z,
            line_name=line_name,
            sfr_model=sfr_model,
            use_scatter=use_scatter,
            params_fisher=params_fisher,
        )

    return L_line



def mhalo_to_lline(
    Mhalo,
    z,
    line_name="CII158",
    model_name="Silva15-m1",
    sfr_model="Behroozi19",
    line_params=inp.line_params_default
):

    if line_name == "CII158":

        L_line = mhalo_to_lcp_fit(
            Mhalo,
            z,
            model_name=model_name,
            sfr_model=sfr_model,
            line_params=line_params
            
        )

    elif line_name[0:2] == "CO":
        L_line = mhalo_to_lco_fit(
            Mhalo, z, line_name=line_name, sfr_model =sfr_model,
            model_name=model_name, f_duty=line_params['f_duty']
        )

    elif line_name == "OIII88":
        L_line = mhalo_to_Oxygen_fit(
            Mhalo,
            z,
            line_name=line_name,
            sfr_model=sfr_model,
            model_name=model_name,
            f_duty=line_params['f_duty'],
        )

    else:
         L_line = L_line_Visbal10(Mhalo, z,
                                  f_duty=0.1,
                                  line_name=line_name,
                                  sfr_model=sfr_model)
    
    
    return L_line


def calc_luminosity(
    boxsize,
    ngrid,
    halocat_file,
    halo_redshift,
    sfr_model="Behroozi19",
    line_name="CII158",
    halo_cutoff_mass=1e11,
    nproj=None,
    use_scatter=False,
    halocat_type="input_cat",
    params_fisher=None,
    zg=8.8
):
    """
    Calculate luminosity grid based on halo properties.

    Parameters:
    -----------
    boxsize : float
        Size of the simulation box in Mpc/h.
    ngrid : int
        Number of grid cells per dimension.
    halocat_file : str
        File path to the halo catalog.
    halo_redshift : float
        Redshift of the halos.
    sfr_model : str, optional
        Star formation rate model (default is "Behroozi19").
    line_name : str, optional
        Name of the spectral line (default is "CII158").
    halo_cutoff_mass : float, optional
        Minimum halo mass in solar masses (default is 1e11).
    nproj : int, optional
        Number of projections (default is None).
    use_scatter : bool, optional
        Whether to use scatter in the model (default is False).
    halocat_type : str, optional
        Type of halo catalog (default is "input_cat").
    params_fisher : dict, optional
        Parameters for Fisher forecast (default is None).
    zg : float, optional
        Redshift of the galaxy formation (default is 8.8).

    Returns:
    --------
    numpy.ndarray
        Luminosity grid calculated based on the specified parameters.
    """
    # Make halocat
    halomass, halo_cm = lu.make_halocat(
        halocat_file, halocat_type=halocat_type, mmin=halo_cutoff_mass, boxsize=boxsize
    )

    # Calculate luminosity line
    lum_line = mhalo_to_lline(
        halomass,
        halo_redshift,
        sfr_model=sfr_model,
        line_name=line_name,
        use_scatter=use_scatter,
        params_fisher=params_fisher,
        zg=zg
    )

    # Make grid
    grid = lu.make_grid(halo_cm, weight=lum_line, boxsize=boxsize, ngrid=ngrid)

    if nproj:
        # Slice grid if projection is required
        grid_lum = lu.slice_2d(grid, boxsize, nproj, operation="sum", axis=2)
    else:
        grid_lum = grid

    return grid_lum



            
class line_modeling:
    def __init__(self, Mhalo, z, line_name="CII158", 
                 model_name="Silva15-m1", 
                 sfr_model="Behroozi19", 
                 line_params=None):
        
        if line_params is None:
            # Assuming line parameters are already imported from input.py outside the class
            
            self.line_params = inp.line_params_default
        
        else:
            # Fill in missing parameters with defaults from input.py
            
            self.line_params = {**inp.line_params_default, **line_params}

        self.line_name = line_name
        self.model_name = model_name
        self.sfr_model = sfr_model
        self.use_scatter = self.line_params['use_scatter']
        self.a_off = self.line_params['a_off']
        self.a_std = self.line_params['a_std']
        self.b_off = self.line_params['b_off']
        self.b_std = self.line_params['b_std']
        self.f_duty = self.line_params['f_duty']
        self.zg = self.line_params['zg']
        self.scatter_dex = self.line_params['scatter_dex']
        self.cov_scatter_matrix = self.line_params['cov_scatter_matrix']
        
        
    def line_luminosity(self, Mhalo, z):
        if not self.use_scatter:
            return mhalo_to_lline(
                Mhalo,
                z,
                line_name=self.line_name,
                model_name=self.model_name,
                sfr_model=self.sfr_model,
                line_params=self.line_params
            )
        
        elif self.use_scatter and self.model_name != "alma_scalling":
            Line_lum = mhalo_to_lline(
                Mhalo,
                z,
                line_name=self.line_name,
                model_name=self.model_name,
                sfr_model=self.sfr_model,
                line_params=self.line_params
                
            )
            line_with_scatter = Line_lum + Line_lum * self.scatter_dex * np.random.normal(0, 1, len(Line_lum))
            return line_with_scatter
        
        elif self.use_scatter and self.model_name == "alma_scalling":
            if self.cov_scatter_matrix is None:
                cov_scatter_matrix = np.array([[self.a_std**2, 0], [0, self.b_std**2]])
            else:
                cov_scatter_matrix = self.cov_scatter_matrix
            
            params_mean = np.array([self.a_off, self.b_off])
            
            params_list = np.random.multivariate_normal(params_mean, cov_scatter_matrix, size=len(Mhalo))
            
            result = []
            for i in range(len(Mhalo)): 
                line_with_scatter = mhalo_to_lline_alma(
                    Mhalo[i],
                    z,
                    line_name=self.line_name,
                    sfr_model=self.sfr_model,
                    line_params={'a_off': params_list[:,0][i], 'b_off': params_list[:,1][i]}
                )
                result.append(line_with_scatter)
            
            return result

            




class theory:
    def __init__(self, cosmo_setup=cosmos.cosmo()):
        
        self.cosmo_setup = cosmo_setup
        
        print("Parameters used in cosmo_setup:")
        print("Hubble constant (h):", self.cosmo_setup.h)
        print("Omega matter (Omega_m):", self.cosmo_setup.omega_m)
        
        
    def N_cen(self, M, M_min=1e8, sigma_logm=0.15):
        res = 0.5 * (1 + erf((np.log10(M) - np.log10(M_min)) / sigma_logm))
        return res

    def N_sat(self, M, M_cut=10**12.23, M1=10**12.75, alpha_g=0.99):
        if np.isscalar(M):
            res = 0.0 if M <= M_cut else self.N_cen(M) * ((M - M_cut) / M1) ** alpha_g
        else:
            below_M_cut = np.where(M < M_cut)[0]
            above_M_cut = np.where(M >= M_cut)[0]
            res = np.zeros(len(M))
            res[below_M_cut] = 0.0
            res[above_M_cut] = self.N_cen(M[above_M_cut]) * ((M[above_M_cut] - M_cut) / M1) ** alpha_g
        return res

    def N_M_hod(self, Mh):
        return (self.N_sat(Mh) + 1) * self.N_cen(Mh)

    def hmf(self, z, HOD_model=False):
        Mass_bin, dndm = self.cosmo_setup.hmf_setup(z)
        if HOD_model:
            return Mass_bin, dndm * self.N_M_hod(Mass_bin)
        else:
            return Mass_bin, dndm

    def I_line(self, z, line_name="CII158", model_name="Silva15-m1", sfr_model="Silva15", HOD_model=False, params_fisher=None):
        mass_bin, dndlnM = self.hmf(z, HOD_model=HOD_model)
        L_line = ll.mhalo_to_lline(mass_bin, z, line_name=line_name, sfr_model=sfr_model, model_name=model_name, params_fisher=params_fisher)
        factor = (inp.c_in_mpc) / (4 * np.pi * inp.nu_rest(line_name=line_name) * self.cosmo_setup.H_z(z))
        conversion_fac = 4.0204e-2  # jy/sr
        integrand = factor * dndlnM * L_line
        integration = simps(integrand, x = np.log(mass_bin))
        return integration * conversion_fac

    def P_shot_gong(self, z, line_name="CII158", sfr_model="Silva15", model_name="Silva15-m1", HOD_model=False, params_fisher=None):
        mass_bin, dndlnM = self.hmf(z, HOD_model=HOD_model)
        L_line = ll.mhalo_to_lline(mass_bin, z, line_name=line_name, sfr_model=sfr_model, model_name=model_name, params_fisher=params_fisher)
        integrand_numerator = dndlnM * L_line**2
        int_numerator = simps(integrand_numerator,  x = np.log(mass_bin))
        factor = (inp.c_in_mpc) / (4 * np.pi * inp.nu_rest(line_name=line_name) * self.cosmo_setup.H_z(z))
        conversion_fac = 4.0204e-2 # jy/sr
        return int_numerator * factor**2 * conversion_fac**2

    def T_line(self, z, line_name="CII158", sfr_model="Silva15", model_name="Silva15-m1", HOD_model=False, params_fisher=None):
        Intensity = self.I_line(z, line_name=line_name, model_name=model_name, sfr_model=sfr_model, HOD_model=HOD_model, params_fisher=params_fisher)
        nu_obs = inp.nu_rest(line_name=line_name) / (1 + z)
        T_line = inp.c_in_mpc**2 * Intensity / (2 * inp.kb_si * nu_obs**2)
        return T_line  # in muK

    def P_shot(self, z, line_name="CII158", sfr_model="Silva15", model_name="Silva15-m1", HOD_model=False, params_fisher=None):
        mass_bin, dndlnM = self.hmf(z, HOD_model=HOD_model)
        L_line = ll.mhalo_to_lline(mass_bin, z, line_name=line_name, sfr_model=sfr_model, model_name=model_name, params_fisher=params_fisher)
        integrand_numerator = dndlnM * L_line**2
        integrand_denominator = dndlnM * L_line
        int_numerator = simps(integrand_numerator, x = np.log(mass_bin))
        int_denominator = simps(integrand_denominator,  x =np.log(mass_bin))
        return int_numerator / int_denominator**2

    def b_line(self, z, line_name="CII158", sfr_model="Silva15", model_name="Silva15-m1", HOD_model=False, params_fisher=None):
        mass_bin, dndlnM = self.hmf(z, HOD_model=HOD_model)
        L_line = ll.mhalo_to_lline(mass_bin, z, line_name=line_name, sfr_model=sfr_model, model_name=model_name, params_fisher=params_fisher)
        integrand_numerator = dndlnM * L_line * self.cosmo_setup.bias_dm(mass_bin, z)
        integrand_denominator = dndlnM * L_line
        int_numerator = simps(integrand_numerator,  x = np.log(mass_bin))
        int_denominator = simps(integrand_denominator, x = np.log(mass_bin))
        return int_numerator / int_denominator
    
    
    
    def Pk_line(self, k, z, line_name="CII158", label="total", sfr_model="Silva15", model_name="Silva15-m1", pk_unit="intensity", HOD_model=False, params_fisher=None):
        if pk_unit == "intensity":
            I_nu_square = (
                self.I_line(
                    z,
                    line_name=line_name,
                    model_name=model_name,
                    sfr_model=sfr_model,
                    HOD_model=HOD_model,
                    params_fisher=params_fisher
                )
                ** 2
            )
            pk_lin = self.cosmo_setup.pk_camb(k, z)

            if label == "total":
                res = I_nu_square * self.b_line(
                    z, line_name=line_name, model_name=model_name, HOD_model=HOD_model, params_fisher=params_fisher
                ) ** 2 * pk_lin + I_nu_square * self.P_shot(
                    z, line_name=line_name, model_name=model_name, HOD_model=HOD_model, params_fisher=params_fisher
                )

            if label == "clustering":
                res = (
                    I_nu_square
                    * self.b_line(
                        z, line_name=line_name, model_name=model_name, HOD_model=HOD_model, params_fisher=params_fisher
                    )
                    ** 2
                    * pk_lin
                )

            if label == "shot":
                res = I_nu_square * self.P_shot(
                    z, line_name=line_name, model_name=model_name, HOD_model=HOD_model, params_fisher=params_fisher
                )

            return res

        if pk_unit == "temperature" or pk_unit == "muk":
            T_line_square = (
                self.T_line(
                    z,
                    line_name=line_name,
                    model_name=model_name,
                    sfr_model=sfr_model,
                    params_fisher=params_fisher
                )
                ** 2
            )
            pk_lin = self.cosmo_setup.pk_camb(k, z)
            if label == "total":
                res = T_line_square * (
                    self.b_line(
                        z, line_name=line_name, model_name=model_name, HOD_model=HOD_model, params_fisher=params_fisher
                    )
                    ** 2
                    * pk_lin
                    + self.P_shot(
                        z, line_name=line_name, model_name=model_name, HOD_model=HOD_model, params_fisher=params_fisher
                    )
                )

            if label == "clustering":
                res = T_line_square * (
                    self.b_line(
                        z,
                        line_name=line_name,
                        sfr_model=sfr_model,
                        model_name=model_name,
                        HOD_model=HOD_model,
                        params_fisher=params_fisher
                    )
                    ** 2
                    * pk_lin
                )

            if label == "shot":
                res = T_line_square * (
                    self.P_shot(
                        z,
                        line_name=line_name,
                        sfr_model=sfr_model,
                        model_name=model_name,
                        HOD_model=HOD_model,
                        params_fisher=params_fisher
                    )
                )
                res = res

            return res
    
    
    
    def window_gauss(self, z, z_mean, deltaz):
        p = (1.0 / np.sqrt(2 * np.pi * deltaz)) * np.exp(
            -((z - z_mean) ** 2) / 2.0 / deltaz**2
        )
        return p

    def Cl_line(
        self,
        ell,
        z,
        deltaz,
        fduty=1.0,
        line_name="CII158",
        label="total",
        sfr_model="Silva15",
        model_name="Silva15-m1",
        pk_unit="temperature",
        params_fisher=None
    ):
        chi = self.cosmo_setup.D_co(z)
        kp = ell / chi
        zint = np.logspace(np.log10(z - deltaz), np.log10(z + deltaz), num=10)
        pk = [
            self.Pk_line(
                kp,
                z,
                line_name=line_name,
                sfr_model=sfr_model,
                model_name=model_name,
                label=label,
                pk_unit=pk_unit,
                params_fisher=params_fisher
            )
            for z in zint
        ]
        integrand = (
            1.0
            / (inp.c_in_mpc)
            * (
                self.window_gauss(zint, z, deltaz)
                * self.cosmo_setup.H_z(z)
                * pk
                / chi ** 2
            )
        )
        res = simps(integrand,  x =zint, axis=1)
        return res
    
    def __init__(self, halocat_file, halo_redshift, sfr_model="Behroozi19",
                 model_name="Silva15-m1", quantity="intensity", line_name="CII158",
                 halo_cutoff_mass=1e11, halocat_type="input_cat", line_params=None, cosmo_params=None,
                 ngrid_x=128, ngrid_y=128, ngrid_z=128, boxsize_x=1, boxsize_y=1, boxsize_z=1,
                 nu_obs=None, dnu_obs=None, theta_fwhm=None, z_evolution=False):

        self.halocat_file = halocat_file
        self.halo_redshift = halo_redshift
        self.sfr_model = sfr_model
        self.model_name = model_name
        self.quantity = quantity
        self.line_name = line_name
        self.halo_cutoff_mass = halo_cutoff_mass    
        self.halocat_type = halocat_type
        self.line_params = line_params
        self.ngrid_x = ngrid_x
        self.ngrid_y = ngrid_y
        self.ngrid_z = ngrid_z
        self.boxsize_x = boxsize_x
        self.boxsize_y = boxsize_y
        self.boxsize_z = boxsize_z
        self.nu_obs = nu_obs
        self.dnu_obs = dnu_obs
        self.theta_fwhm = theta_fwhm
        self.z_evolution = z_evolution
        
        if cosmo_params is None:
            self.cosmo_setup = cosmos.cosmo()
        else:
            self.cosmo_setup = cosmos.cosmo(cosmo_params)
            
        self.small_h = self.cosmo_setup.h
        
        
            
    def make_quantity_grid(self):
        # Calculate cell size
        cellsize_x = self.boxsize_x / self.ngrid_x
        #cellsize_y = self.boxsize_y / self.ngrid_y
        cellsize_z = self.boxsize_z / self.ngrid_z
        
        # Calculate solid angle per pixel
        d_omega_pix = self.cosmo_setup.solid_angle(cellsize_x, self.halo_redshift)
        
        # Calculate comoving size to delta nu conversion factor
        d_nu = self.cosmo_setup.comoving_size_to_delta_nu(cellsize_z, self.halo_redshift, line_name=self.line_name)
        
        # Make halocat
        halomass, halo_cm = lu.make_halocat(
            self.halocat_file, halocat_type=self.halocat_type, mmin=self.halo_cutoff_mass, boxsize=self.boxsize_x
        )
        
        # Initialize LineConverter
        line_converter = line_modeling(
            halomass, self.halo_redshift, line_name=self.line_name, model_name=self.model_name,
            sfr_model=self.sfr_model, line_params=self.line_params
        )
        
        # Calculate luminosity or intensity line
        lum_line = line_converter.line_luminosity(halomass, self.halo_redshift)
        
        # Calculate intensity grid
        grid_lum = lu.make_grid_rectangular(
            halo_cm,
            weight=lum_line,
            ngrid_x=self.ngrid_x,
            ngrid_y=self.ngrid_y,
            ngrid_z=self.ngrid_z,
            boxsize_x=self.boxsize_x,
            boxsize_y=self.boxsize_y,
            boxsize_z=self.boxsize_z,
        )
        
        if self.quantity.lower() == "luminosity":
            return grid_lum
        elif self.quantity.lower() == "intensity":
            # Calculate intensity grid
            D_lum = self.cosmo_setup.D_luminosity(self.halo_redshift)
            prefac = 4.0204e-2 * self.small_h ** 2  # Lsol/Mpc^2/ GHz
            grid_intensity = (
                prefac * (grid_lum / 4.0 / np.pi / D_lum ** 2) / d_omega_pix / d_nu
            )  # Jy/sr
            return grid_intensity
        else:
            raise ValueError("Invalid quantity specified. Must be 'luminosity' or 'intensity'.")

    def make_quantity_rectangular_grid_no_z_evo(self):
        if self.dnu_obs is not None:
            zem, dz, dchi, d_ngrid = self.cosmo_setup.box_freq_to_quantities(
                nu_obs=self.nu_obs,
                dnu_obs=self.dnu_obs,
                boxsize=self.boxsize_z,
                ngrid=self.ngrid_z,
                z_start=self.halo_redshift,
                line_name=self.line_name
            )
            Ngrid_new = min(int(self.ngrid_z / d_ngrid), 1)
            d_ngrid = min(d_ngrid, self.ngrid_z)
        else:
            Ngrid_new = self.ngrid_z
            d_ngrid = self.ngrid_z

        self.ngrid_z = Ngrid_new
        Igcal = self.make_quantity_grid()

        return Igcal
    
    
    
    
    def make_quantity_rectangular_grid_z_evo(self):
        if self.dnu_obs is not None:
            zem, dz, dchi, d_ngrid = self.cosmo_setup.box_freq_to_quantities(
                nu_obs=self.nu_obs,
                dnu_obs=self.dnu_obs,
                boxsize=self.boxsize_z,
                ngrid=self.ngrid_z,
                z_start=self.halo_redshift,
                line_name=self.line_name
            )
            Ngrid_new = int(self.ngrid_z / d_ngrid) if d_ngrid < self.ngrid_z else 1
            d_ngrid = d_ngrid if d_ngrid < self.ngrid_z else self.ngrid_z

        # if dnu_obs is None, then ngrid along z axis will remain unchanged.
        if self.dnu_obs is None:
            Ngrid_new = self.ngrid_z
            d_ngrid = self.ngrid_z

        cellsize_x = self.boxsize_x / self.ngrid_x
        cellsize_y = self.boxsize_y / self.ngrid_y
        cellsize_z = self.boxsize_z / Ngrid_new

        halomass, halo_cm = lu.make_halocat(
            self.halocat_file, mmin=self.halo_cutoff_mass, halocat_type=self.halocat_type
        )

        grid_intensity = np.array([])

        for i in range(Ngrid_new):
            z_start = self.halo_redshift + (i * dz)

            if cellsize_x:
                d_omega_pix = self.cosmo_setup.solid_angle(cellsize_x, z_start)

            if cellsize_y:
                d_omega_pix = self.cosmo_setup.solid_angle(cellsize_y, z_start)

            d_nu = self.cosmo_setup.comoving_size_to_delta_nu(cellsize_z, z_start, line_name=self.line_name)

            zrange = halo_cm[:, 2]
            mask = np.where((zrange >= i * dchi) & (zrange < (i + 1) * dchi))

            x_mask = halo_cm[:, 0][mask]
            y_mask = halo_cm[:, 1][mask]
            z_mask = zrange[mask]


            hloc_mask = np.stack((x_mask, y_mask, z_mask), axis=1)

            # Initialize LineConverter
            line_converter = line_modeling(
                halomass, self.halo_redshift, line_name=self.line_name, model_name=self.model_name,
                sfr_model=self.sfr_model, line_params=self.line_params
            )
            
            # Calculate luminosity or intensity line
            lum_line = line_converter.line_luminosity(halomass, self.halo_redshift)

            grid_lum = lu.make_grid_rectangular(
                hloc_mask,
                weight=lum_line,
                ngrid_x=self.ngrid_x,
                ngrid_y=self.ngrid_y,
                ngrid_z=1,
                boxsize_x=self.boxsize_x,
                boxsize_y=self.boxsize_y,
                boxsize_z=self.boxsize_z,
            )

            D_lum = self.cosmo_setup.D_luminosity(self.halo_redshift)
            prefac = 4.0204e-2 * self.small_h ** 2  # Lsol/Mpc^2/ GHz

            grid_intensity_cal = (
                    prefac * (grid_lum / 4.0 / np.pi / D_lum ** 2) / d_omega_pix / d_nu
            )  # Jy/sr

            if i == 0:
                grid_intensity = grid_intensity_cal
            else:
                grid_intensity = np.dstack([grid_intensity, grid_intensity_cal])

        return grid_intensity
    
    
        
    def make_intensity_grid(self):
        if not self.z_evolution:
            Igcal = self.make_quantity_rectangular_grid_no_z_evo()
        else:
            Igcal = self.make_quantity_rectangular_grid_z_evo()

        if self.theta_fwhm is None:
            return Igcal

        elif self.theta_fwhm:
            Ngrid_z = np.shape(Igcal)[2]
            zem, dz, dchi, d_ngrid = self.cosmo_setup.box_freq_to_quantities(nu_obs=self.nu_obs,
                                                                              dnu_obs=self.dnu_obs,
                                                                              boxsize=self.boxsize_z,
                                                                              ngrid=self.ngrid_z,
                                                                              line_name=self.line_name)

            convolved_grid = []
            theta = lu.convert_beam_unit_to_radian(self.theta_fwhm, beam_unit='arcmin')
            cellsize_x = self.boxsize_x / self.ngrid_x
            cellsize_y = self.boxsize_y / self.ngrid_y

            for i in range(Ngrid_z):
                z_start = self.halo_redshift + (i * dz)
                beam_size = self.cosmo_setup.angle_to_comoving_size(z_start, theta)
                beam_std_x = beam_size / (np.sqrt(8 * np.log(2.0))) / cellsize_x
                beam_std_y = beam_size / (np.sqrt(8 * np.log(2.0))) / cellsize_y

                gauss_kernel = Gaussian2DKernel(beam_std_x, y_stddev=beam_std_y)
                grid_quantity = Igcal[:, :, i: i + 1].reshape(self.ngrid_x, self.ngrid_y)
                convolved_grid_cal = convolve(grid_quantity, gauss_kernel)
                convolved_grid.append(convolved_grid_cal)

            Igcal = np.swapaxes(convolved_grid, 0, 2)

        return Igcal
    
    

    def get_beam_cov_3d(self, grid_quantity):
        
        theta = lu.convert_beam_unit_to_radian(self.theta_fwhm, beam_unit=self.beam_unit)
        beam_size = self.cosmo_setup.angle_to_comoving_size(self.halo_redshift, theta)
        
        cellsize_x = self.boxsize_x / self.ngrid_x
        cellsize_y = self.boxsize_y / self.ngrid_y
        
        beam_std_x = beam_size / (np.sqrt(8 * np.log(2.0))) / cellsize_x
        beam_std_y = beam_size / (np.sqrt(8 * np.log(2.0))) / cellsize_y
    
        gauss_kernel = Gaussian2DKernel(beam_std_x, y_stddev=beam_std_y)
    
        convolved_grid = []
    
        for i in range(self.Ngrid_new):
            grid_start = i * self.d_ngrid
            grid_end = (i + 1) * self.d_ngrid
            grid_quantity1 = np.mean(grid_quantity[:, :, grid_start:grid_end], axis=2)
            convolved_grid_cal = convolve(grid_quantity1, gauss_kernel)
            convolved_grid.append(convolved_grid_cal)
    
        return np.swapaxes(convolved_grid, 0, 2)
    
    
    
    def make_grid_dnu_obs(self, grid):
        global zem, dz, dchi, d_ngrid 
        
        # Compute useful quantities
        zem, dz, dchi, d_ngrid = self.cosmo_setup.box_freq_to_quantities(nu_obs=self.nu_obs,
                                                                          dnu_obs=self.dnu_obs,
                                                                          boxsize=self.boxsize_z,
                                                                          ngrid=self.ngrid_z,
                                                                          line_name=self.line_name)

        Ngrid_new = int(self.ngrid_z / d_ngrid) if d_ngrid < self.ngrid_z else 1
        d_ngrid = min(d_ngrid, self.ngrid_z)
        
        if self.theta_fwhm is not None:
            convolved_grid = []
            theta = lu.convert_beam_unit_to_radian(self.theta_fwhm, beam_unit='arcmin')
            
            for i in range(Ngrid_new):
                grid_start = i * d_ngrid
                grid_end = (i + 1) * d_ngrid
                z_start = zem + (i * dz)
                beam_size = self.cosmo_setup.angle_to_comoving_size(z_start, theta)
                beam_std = beam_size / (np.sqrt(8 * np.log(2.0)))
                gauss_kernel = Gaussian2DKernel(beam_std)
                grid_quantity = np.mean(grid[:, :, grid_start: grid_end], axis=2)
                convolved_grid_cal = convolve(grid_quantity, gauss_kernel)
                convolved_grid.append(convolved_grid_cal)

            Igcal = np.swapaxes(convolved_grid, 0, 2)

            return Igcal

        if self.theta_fwhm is None:
            grid_split = np.split(grid, Ngrid_new, axis=2)
            grid_mean = np.mean(grid_split, axis=3)
            Igcal = np.swapaxes(grid_mean, 0, 2)

            return Igcal



