#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division

import matplotlib as pl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from astropy.modeling.models import Gaussian2D
from scipy.interpolate import RectBivariateSpline

import limpy.params as p
import limpy.utils as lu

pl.rcParams["xtick.labelsize"] = "10"
pl.rcParams["ytick.labelsize"] = "10"
pl.rcParams["axes.labelsize"] = "15"
pl.rcParams["axes.labelsize"] = "15"

small_h = p.cosmo.h

################################################################################
##     Interpolate the sfr for different models at the beginning
################################################################################

data_path = "../data/"

# Read the files of saved data files for sfr
sfr_file_tng100 = data_path + "sfr_processed_TNG100-1.npz"
sfr_file_tng300 = data_path + "sfr_processed_TNG300-1.npz"
sfr_file_Behroozi19 = data_path + "sfr_Behroozi.dat"


# read and interpolate Behroozi
z, m, sfr_file = np.loadtxt(sfr_file_Behroozi19, unpack=True)
zlen = 137  # manually checked
mlen = int(len(z) / zlen)
zn = z[0:zlen]
log_mh = np.log10(m.reshape(mlen, zlen)[:, 0] / small_h)
sfr_int = sfr_file.reshape(mlen, zlen)

sfr_interpolation_Behroozi19 = RectBivariateSpline(log_mh, zn, sfr_int)

# TNG 100
f = np.load(sfr_file_tng100)
sfr_tng100 = f["sfr"]
z_tng100 = f["z"]
log_mass_tng100 = f["halomass"]
sfr_interpolation_tng100 = RectBivariateSpline(log_mass_tng100, z_tng100, sfr_tng100)

# TNG 300
f = np.load(sfr_file_tng300)
sfr_tng300 = f["sfr"]
z_tng300 = f["z"]
log_mass_tng300 = f["halomass"]
sfr_interpolation_tng300 = RectBivariateSpline(log_mass_tng300, z_tng300, sfr_tng300)


################################################################################
##      SFR and line luminosity models
################################################################################

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
    if sfr_model != None:
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    else:
        fstar = 0.1
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * (fstar / f_duty)
    return L


def L_CO_Visbal10(M, z, f_duty=0.1, J_ladder=1, sfr_model=None):
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

    if sfr_model != None:
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    else:
        fstar = 0.1
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * (fstar / f_duty)

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

    if sfr_model != None:
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    else:
        fstar = 0.1
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * (fstar / f_duty)

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

    if sfr_model != None:
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    else:
        fstar = 0.1
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * (fstar / f_duty)
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

    if sfr_model != None:
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    else:
        fstar = 0.1
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * (fstar / f_duty)
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

    fstar = 0.1
    R = 4.8e4

    if sfr_model != None:
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    else:
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * (fstar / f_duty)
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

    fstar = 0.1
    R = 2.3e6

    if sfr_model != None:
        sfr_out = mhalo_to_sfr(M, z, sfr_model=sfr_model)
        L = R * sfr_out

    else:
        L = 6.6e6 * (R / 3.8e6) * (M / 1e10) * ((1 + z) / 7) ** 1.5 * (fstar / f_duty)
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
    L = sfr_out / factor / p.Lsun_erg_s
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

    L = (factor / p.Lsun_erg_s) * (sfr_out) ** gamma
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


def L_line_Visbal10(M, z, f_duty=0.1, line_name="CO10", sfr_model=None):
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

    if line_name in p.line_list:
        pass
    else:
        print(
            "line name is not found! Choose a line name from the list:\n", p.line_list
        )

    if line_name[0:2] == "CO":
        line_name_len = len(line_name)
        if line_name_len == 4:
            J_ladder = int(line_name[2])
        elif line_name_len == 5 or line_name_len == 6:
            J_ladder = int(line_name[2:4])

        L_line = L_CO_Visbal10(
            M, z, f_duty=f_duty, sfr_model=sfr_model, J_ladder=J_ladder
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


def read_sfr_lowm(SFR_filename):
    """
    This takes a SFR file in Universemachine Output format

    Parameters:
        filename: SFR filename/filepath

    Returns:
        redshift, SFR, error_SFR_up, error_SFR_down

    Note: Columns are: Scale factor, SFR, Err_Up, Err_Down (all linear units)
    """

    # read SFR parameters from SFR file generated from Universe machine
    onepz, SFR = np.loadtxt(SFR_filename, unpack=True, usecols=(0, 2))

    # scale factor to redshift
    z = onepz - 1

    return z, 10**SFR


def read_sfr_highm(SFR_filename):
    """
    This takes a SFR file in Universemachine Output format

    Parameters:
        filename: SFR filename/filepath

    Returns:
        redshift, SFR, error_SFR_up, error_SFR_down

    Note: Columns are: Scale factor, SFR, Err_Up, Err_Down (all linear units)
    """
    # read SFR parameters from SFR file generated from Universe machine
    scale_fac, SFR, error_SFR_up, error_SFR_down = np.loadtxt(SFR_filename, unpack=True)
    # scale factor to redshift
    z = 1.0 / scale_fac - 1.0

    return z, SFR, error_SFR_up, error_SFR_down


def make_hlist_ascii_to_npz(hlist_path_ascii, saved_filename=None):
    """
    Takes a hlist file in the form of Universemachine format and tranforms
    to a npz file with necessary quantities.
    coumn 0: scale factor
    coulumn 1: halo mass
    column 2: x co-ordinate of halos
    column 3: y co-ordinate of halos
    column 4: z co-ordinate of halos
    """

    # reads only scale factor, halomass,x,y and z from the ascii file used in Universe machine
    data = np.loadtxt(hlist_path_ascii, usecols=(0, 10, 17, 18, 19))

    # save the file in npz format either in mentioned filename or in original ascii filename
    if saved_filename:
        np.savez(
            a=data[:, 0],
            m=data[:, 1],
            x=data[:, 2],
            y=data[:, 3],
            z=data[:, 4],
        )
    else:
        np.savez(
            hlist_path_ascii,
            a=data[:, 0],
            m=data[:, 1],
            x=data[:, 2],
            y=data[:, 3],
            z=data[:, 4],
        )
    return


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

    Mhalo /= p.cosmo.h
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
    factor = (1.4 - 0.07 * z) * np.log10(sfr_out) + 7.1 - 0.07 * z
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
    Lcii = F_z * ((Mhalo / M1) ** beta) * np.exp(-N1 / Mhalo)
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


def sfr_to_L_line_alma(
    z, sfr, line_name="CII158", use_scatter=False, params_fisher=None
):
    """
    Calculates lumiosity of the OIII lines from SFR assuming a 3\sigma Gussian
    scatter. The parameter values for the scattered relation
    is mentioned in defalut_params module.

    Input: z and sfr

    return: luminosity of OIII lines in log scale
    """

    if line_name in p.line_list:
        pass
    else:
        assert "Not a familiar line."

    if params_fisher == None:
        a_off, a_std, b_off, b_std = p.line_scattered_params(line_name)

    else:
        a_off, a_std, b_off, b_std = p.line_scattered_params(line_name)
        params_list_default = {
            "a_off": a_off,
            "a_std": a_std,
            "b_off": b_off,
            "b_std": b_std,
        }
        params_list_new = lu.update_params(params_list_default, params_fisher)

        a_off = params_list_new["a_off"]
        a_std = params_list_new["a_std"]
        b_off = params_list_new["b_off"]
        b_std = params_list_new["b_std"]
        # print("included params_fisher")

    def L_co_log(sfr, alpha, beta):
        nu_co_line = p.nu_rest(line_name)
        L_ir_sun = sfr * 1e10
        L_coprime = (L_ir_sun * 10 ** (-beta)) ** (1 / alpha)
        L_co = 4.9e-5 * (nu_co_line / 115.27) ** 3 * L_coprime
        return np.log10(L_co)

    if np.isscalar(sfr) == True:
        sfr = np.atleast_1d(
            sfr
        )  ####Convert inputs to arrays with at least one dimension.
        ##Scalar inputs are converted to 1-dimensional arrays,
        ##whilst higher-dimensional inputs are preserved.

    sfr_len = len(sfr)
    log_L_line = np.zeros(sfr_len)

    if use_scatter == True:
        if line_name == "CII158" or line_name == "OIII88":
            for i in range(sfr_len):
                a = np.random.normal(a_off, a_std)
                b = np.random.normal(b_off, b_std)
                log_L_line[i] = a + b * np.log10(sfr[i])

        elif line_name[0:2] == "CO":
            for i in range(sfr_len):
                a = np.random.normal(a_off, a_std)
                b = np.random.normal(b_off, b_std)

                log_L_line[i] = L_co_log(sfr[i], a, b)

        return 10**log_L_line

    if use_scatter == False:
        for i in range(sfr_len):
            if line_name == "CII158" or line_name == "OIII88":
                log_L_line[i] = a_off + b_off * np.log10(sfr[i])

            elif line_name[0:2] == "CO":
                log_L_line[i] = L_co_log(sfr[i], a_off, b_off)

        return 10**log_L_line


def mhalo_to_sfr(Mhalo, z, sfr_model="Behroozi19"):
    """
    Returns the SFR history for discrete values of halo mass.
    """

    if sfr_model == "Silva15":
        sfr = sfr_silva15(Mhalo, z)

    if sfr_model == "Fonseca16":
        sfr = sfr_Fonseca16(Mhalo, z)

    if sfr_model == "Behroozi19":
        # sfr_interpolation=RectBivariateSpline(mhn, zn, sfrn)
        sfr = sfr_interpolation_Behroozi19(np.log10(Mhalo), z)

    if sfr_model == "Tng100":

        sfr = sfr_interpolation_tng100(np.log10(Mhalo), z)

    if sfr_model == "Tng300":
        sfr = sfr_interpolation_tng300(np.log10(Mhalo), z)

    res = np.where(sfr < 1e-4, p.lcp_low, sfr)
    return res.flatten()


def mhalo_to_lline_alma(
    Mh,
    z,
    line_name="CII158",
    sfr_model="Behroozi19",
    use_scatter=False,
    params_fisher=None,
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
        use_scatter=use_scatter,
        params_fisher=params_fisher,
    )

    return L_line


def mhalo_to_lcp_fit(
    Mhalo,
    z,
    model_name="Silva15-m1",
    sfr_model="Behroozi19",
    use_scatter=False,
    params_fisher=None,
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

    if model_name == "Schaerer20":
        LCII = LCII_Schaerer20(Mhalo, z, sfr_model=sfr_model)

    if model_name == "Alma_scalling":
        LCII = mhalo_to_lline_alma(
            Mhalo,
            z,
            line_name="CII158",
            sfr_model=sfr_model,
            use_scatter=use_scatter,
            params_fisher=params_fisher,
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
        L_line = L_line_Visbal10(Mhalo, z, f_duty=f_duty, line_name=line_name)

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
    use_scatter=False,
    params_fisher=None,
    f_duty=0.1,
):

    if line_name == "CII158":

        L_line = mhalo_to_lcp_fit(
            Mhalo,
            z,
            model_name=model_name,
            sfr_model=sfr_model,
            use_scatter=use_scatter,
            params_fisher=params_fisher,
        )

    elif line_name[0:2] == "CO":
        L_line = mhalo_to_lco_fit(
            Mhalo, z, line_name=line_name, sfr_model =sfr_model,
            model_name=model_name, f_duty=f_duty
        )

    elif line_name == "OIII88":
        L_line = mhalo_to_Oxygen_fit(
            Mhalo,
            z,
            line_name=line_name,
            sfr_model=sfr_model,
            model_name=model_name,
            f_duty=f_duty,
        )

    else:
         L_line = L_line_Visbal10(Mhalo, z,
                                  f_duty=0.1,
                                  line_name=line_name,
                                  sfr_model=sfr_model)
    return L_line


def slice(datacube, ngrid, nproj, option="C"):
    """
    Produces a slice from a 3D data cube for plotting. `option' controls
    whether data cube has C or Fortran ordering.

    """
    iarr = np.zeros(ngrid * ngrid)
    jarr = np.zeros(ngrid * ngrid)
    valarr = np.zeros(ngrid * ngrid)

    counter = 0
    for i in range(ngrid):
        for j in range(ngrid):
            iarr[counter] = i
            jarr[counter] = j
            valarr[counter] = 0.0
            for k in range(nproj):
                if option == "F":
                    valarr[counter] += datacube[i + ngrid * (j + ngrid * k)]
                elif option == "C":
                    valarr[counter] += datacube[k + ngrid * (j + ngrid * i)]
            counter += 1

    return iarr, jarr, valarr


def plot_slice(
    boxsize,
    ngrid,
    nproj,
    dens_gas_file,
    halocat_file,
    halo_redshift,
    line_name="CII158",
    halocat_type="input_cat",
    halo_cutoff_mass=1e11,
    use_scatter=True,
    density_plot=False,
    halo_overplot=False,
    plot_lines=False,
    unit="mpc",
    line_cb_min=1e2,
    line_cb_max=1e10,
    params_fisher=None,
):

    """
    Plot a slice of gas density field and overplot the distribution of
    haloes in that slice.

    boxsize: size of box in Mpc
    ngrid: number of grids along the one axis of simulation box
    nproj: cells to project (values could range from 1 to the ngrid)
    halocat_file: path to halo catalogue file (full path)
    halo_redshift: redshift of halos
    halo_cutoff_mass: cutt off mass of the halos
    density_plot: If true, plot the density distribution
    halo_plot: If True, plot halos

    tick_label=either 'mpc' or degree

    """

    low_mass_log = 0.0
    # z_halos=6.9  #Calculated from the scale factor of halo catalouge

    cellsize = boxsize / ngrid

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    if density_plot:

        # Load density file
        with open(dens_gas_file, "rb") as f:
            dens_gas = np.fromfile(f, dtype="f", count=-1)
            # dens_gas=dens_gas+1.0
            # dens_gas=dens_gas.reshape(ngrid,ngrid,ngrid)
            # rhobar = np.mean(dens_gas)

        # slice the data cube
        i, j, val = slice(dens_gas, ngrid, nproj)

        dens_mean = np.mean(dens_gas + 1)

        val = val / (dens_mean * nproj)

        cellsize = boxsize / ngrid
        i *= cellsize
        j *= cellsize

        s = plt.scatter(
            i,
            j,
            c=val,
            s=10,
            marker="s",
            edgecolor="none",
            rasterized=True,
            cmap="magma",
            vmax=0.5,
            vmin=-0.5,
        )

        if unit == "mpc":
            ax.set_xlim(0, boxsize)
            ax.set_ylim(0, boxsize)
            plt.xlabel("cMpc/h")
            plt.ylabel("cMpc/h")

        elif unit == "degree":
            ax.set_xlim(0, boxsize)
            ax.set_ylim(0, boxsize)

            xmin = 0
            ymin = 0
            xmax = ymax = lu.comoving_boxsize_to_degree(halo_redshift, boxsize)

            N = 4
            xtick_mpc = ytick_mpc = np.linspace(0, boxsize, N)

            custom_yticks = np.round(np.linspace(ymin, ymax, N, dtype=float), 1)

            ax.set_yticks(ytick_mpc)
            ax.set_yticklabels(custom_yticks)

            custom_xticks = np.round(np.linspace(xmin, xmax, N, dtype=float), 1)
            ax.set_xticks(xtick_mpc)
            ax.set_xticklabels(custom_xticks)

            plt.xlabel(r"$\Theta\,(\mathrm{degree})$")
            plt.ylabel(r"$\Theta\,(\mathrm{degree})$")

        divider = make_axes_locatable(ax)

        cax = divider.append_axes("bottom", "3%", pad="13%")
        cb = plt.colorbar(s, cax=cax, orientation="horizontal")
        cb.set_label(r"$\Delta_\rho$", labelpad=5)

        cb.solids.set_edgecolor("face")
        ax.set_aspect("equal", "box")

    if halo_overplot:

        # Load density file
        with open(dens_gas_file, "rb") as f:
            dens_gas = np.fromfile(f, dtype="f", count=-1)
            # dens_gas=dens_gas+1.0
            # dens_gas=dens_gas.reshape(ngrid,ngrid,ngrid)
            # rhobar = np.mean(dens_gas)

        # slice the data cube
        i, j, val = slice(dens_gas, ngrid, nproj)

        dens_mean = np.mean(dens_gas + 1)

        val = val / (dens_mean * nproj)

        cellsize = boxsize / ngrid
        i *= cellsize
        j *= cellsize

        s = plt.scatter(
            i,
            j,
            c=val,
            s=10,
            marker="s",
            edgecolor="none",
            rasterized=True,
            cmap=plt.cm.viridis_r,
            vmax=1,
            vmin=-1,
        )

        halomass, halo_cm = lu.make_halocat(
            halocat_file,
            mmin=halo_cutoff_mass,
            halocat_type=halocat_type,
            boxsize=boxsize,
        )

        nhalo = len(halomass)
        # Overplot halos
        x_halos = halo_cm[range(0, nhalo * 3, 3)]
        y_halos = halo_cm[range(1, nhalo * 3, 3)]
        z_halos = halo_cm[range(2, nhalo * 3, 3)]

        print("Minimum halo mass:", halomass.min())
        print("Maximum halo mass:", halomass.max())

        # halomass_filter=halomass
        # logmh=np.log10(halomass)
        # logmh=np.array([int(logmh[key]) for key in range(nhalo)])

        # highmass_filter=np.where(logmh>halo_cutoff_mass,logmh,low_mass_log)

        highmass_filter = np.where(halomass > halo_cutoff_mass, halomass, low_mass_log)

        # z_min = 0.0
        z_max = nproj * cellsize  # See slice() above

        mask = z_halos < z_max
        x_halos = x_halos[mask]
        y_halos = y_halos[mask]
        r = highmass_filter[mask]
        r = r / r.max()

        s1 = plt.scatter(
            x_halos, y_halos, marker="o", s=100 * r, color="red", alpha=0.9
        )

        if unit == "mpc":
            ax.set_xlim(0, boxsize)
            ax.set_ylim(0, boxsize)
            plt.xlabel("cMpc/h")
            plt.ylabel("cMpc/h")

        elif unit == "degree":
            ax.set_xlim(0, boxsize)
            ax.set_ylim(0, boxsize)

            xmin = 0
            ymin = 0
            xmax = ymax = lu.comoving_boxsize_to_degree(halo_redshift, boxsize)

            N = 4
            xtick_mpc = ytick_mpc = np.linspace(0, boxsize, N)

            custom_yticks = np.round(np.linspace(ymin, ymax, N, dtype=float), 1)

            ax.set_yticks(ytick_mpc)
            ax.set_yticklabels(custom_yticks)

            custom_xticks = np.round(np.linspace(xmin, xmax, N, dtype=float), 1)
            ax.set_xticks(xtick_mpc)
            ax.set_xticklabels(custom_xticks)

            plt.xlabel(r"$\Theta\,(\mathrm{degree})$")
            plt.ylabel(r"$\Theta\,(\mathrm{degree})$")

        divider = make_axes_locatable(ax)

        cax = divider.append_axes("bottom", "3%", pad="13%")
        cb = plt.colorbar(s, cax=cax, orientation="horizontal")
        cb.set_label(r"$\Delta_\rho$", labelpad=5)

        cb.solids.set_edgecolor("face")
        ax.set_aspect("equal", "box")

    if plot_lines:

        """
        Plot a slice of gas density field and overplot the distribution of
        haloes in that slice.

        """

        with open(dens_gas_file, "rb") as f:
            dens_gas = np.fromfile(f, dtype="f", count=-1)
            # dens_gas=dens_gas+1.0
            # dens_gas=dens_gas.reshape(ngrid**3,)
            # rhobar = np.mean(dens_gas)

        # Plot gas density
        i, j, val = slice(dens_gas, ngrid, nproj)
        # val = val/(rhobar*nproj)

        cellsize = boxsize / ngrid
        i *= cellsize
        j *= cellsize

        dens_mean = np.mean(dens_gas + 1)

        val = val / (dens_mean * nproj)

        s = plt.scatter(
            i,
            j,
            c=val,
            s=10,
            marker="s",
            edgecolor="none",
            rasterized=False,
            cmap="viridis_r",
            vmax=1,
            vmin=-1,
            alpha=0.9,
        )

        xl, yl, lum = calc_luminosity(
            boxsize,
            ngrid,
            nproj,
            halocat_file,
            halo_redshift,
            line_name=line_name,
            halo_cutoff_mass=halo_cutoff_mass,
            halocat_type=halocat_type,
            use_scatter=use_scatter,
            unit="mpc",
        )

        r = (np.log10(lum) / np.log10(lum.max())) ** 6

        s1 = plt.scatter(
            xl,
            yl,
            marker="o",
            c=lum,
            s=70 * r,
            cmap="afmhot",
            vmin=line_cb_min,
            vmax=line_cb_max,
            norm=colors.LogNorm(),
            alpha=0.9,
        )

        if unit == "mpc":
            ax.set_xlim(0, boxsize)
            ax.set_ylim(0, boxsize)
            plt.xlabel("cMpc")
            plt.ylabel("cMpc")

        elif unit == "degree":
            ax.set_xlim(0, boxsize)
            ax.set_ylim(0, boxsize)

            xmin = 0
            ymin = 0
            xmax = ymax = lu.comoving_boxsize_to_degree(halo_redshift, boxsize)

            N = 4
            xtick_mpc = ytick_mpc = np.linspace(0, boxsize, N)

            custom_yticks = np.round(np.linspace(ymin, ymax, N, dtype=float), 1)

            ax.set_yticks(ytick_mpc)
            ax.set_yticklabels(custom_yticks)

            custom_xticks = np.round(np.linspace(xmin, xmax, N, dtype=float), 1)
            ax.set_xticks(xtick_mpc)
            ax.set_xticklabels(custom_xticks)

            plt.xlabel(r"$\Theta\,(\mathrm{degree})$")
            plt.ylabel(r"$\Theta\,(\mathrm{degree})$")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(s1, cax=cax)
        cb.set_label(r"$L_{\mathrm{%s}}\,[L_\odot]$" % (line_name), labelpad=1)

        cb.solids.set_edgecolor("face")
        ax.set_aspect("equal", "box")

        cax1 = divider.append_axes("bottom", "3%", pad="13%")
        cb1 = plt.colorbar(s, cax=cax1, orientation="horizontal")
        cb1.set_label(r"$\Delta_\rho$", labelpad=5)
        cb1.solids.set_edgecolor("face")

    plt.tight_layout()
    plt.savefig("slice_plot.pdf", bbox_inches="tight")


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
):
    halomass, halo_cm = lu.make_halocat(
        halocat_file, halocat_type=halocat_type, mmin=halo_cutoff_mass, boxsize=boxsize
    )

    lum_line = mhalo_to_lline(
        halomass,
        halo_redshift,
        sfr_model=sfr_model,
        line_name=line_name,
        use_scatter=use_scatter,
        params_fisher=params_fisher,
    )

    grid = lu.make_grid(halo_cm, weight=lum_line, boxsize=boxsize, ngrid=ngrid)

    if nproj:
        grid_lum = lu.slice_2d(grid, boxsize, nproj, operation="sum", axis=2)

    else:
        grid_lum = grid

    return grid_lum


def calc_quantity(
    halomass,
    halo_redshift,
    sfr_model="Behroozi19",
    quantity="intensity",
    line_name="CII158",
    halo_cutoff_mass=1e11,
    nproj=None,
    use_scatter=False,
    params_fisher=None,
):

    lum_line = mhalo_to_lline(
        halomass,
        halo_redshift,
        sfr_model=sfr_model,
        use_scatter=use_scatter,
        line_name=line_name,
        params_fisher=params_fisher,
    )

    if quantity == "intensity":
        I_line = (
            (
                lum_line
                / 4.0
                / np.pi
                / p.cosmo.D_luminosity(halo_redshift) ** 2
                / (1 + halo_redshift) ** 2
            )
            * (p.Lsun / (p.jy_unit))
            / (4 * np.pi)
        )

    if quantity == "intensity":
        return I_line
    if quantity == "luminosity":
        return lum_line


def make_quantity_grid(
    boxsize,
    ngrid,
    halocat_file,
    halo_redshift,
    sfr_model="Behroozi19",
    model_name="Silva15-m1",
    quantity="intensity",
    line_name="CII158",
    halo_cutoff_mass=1e11,
    nproj=None,
    use_scatter=False,
    halocat_type="input_cat",
    params_fisher=None,
):
    cellsize = boxsize / ngrid
    # nu_line=p.nu_rest(line_name=line_name)*p.Ghz_to_hz
    d_omega_pix = lu.solid_angle(cellsize, halo_redshift)

    d_nu = lu.comoving_size_to_delta_nu(cellsize, halo_redshift, line_name=line_name)

    # Vcell=(cellsize*p.mpc_to_m)**3

    halomass, halo_cm = lu.make_halocat(
        halocat_file, halocat_type=halocat_type, mmin=halo_cutoff_mass, boxsize=boxsize
    )

    lum_line = mhalo_to_lline(
        halomass,
        halo_redshift,
        sfr_model=sfr_model,
        model_name=model_name,
        use_scatter=use_scatter,
        line_name=line_name,
        params_fisher=params_fisher,
    )
    # if(quantity=='intensity' or quantity=='Inetnsity' or quantity=='I'):
    #    print("debugging")

    if quantity == "luminosity" or quantity == "Luminosity" or quantity == "L":
        grid_lum = lu.make_grid(
            halo_cm, output_grid_dim="3D", weight=lum_line, boxsize=boxsize, ngrid=ngrid
        )

        return grid_lum

    if quantity == "intensity":
        grid_lum = lu.make_grid(
            halo_cm, output_grid_dim="3D", weight=lum_line, boxsize=boxsize, ngrid=ngrid
        )


        D_lum = p.cosmo.D_luminosity(halo_redshift)
        prefac = 4.0204e-2 * small_h**2  # Lsol/Mpc^2/ GHz

        grid_intensity = (
            prefac * (grid_lum / 4.0 / np.pi / D_lum**2) / d_omega_pix / d_nu
        )  # Jy/sr

        return grid_intensity


def make_quantity_rectangular_grid(
    halocat_file,
    halo_redshift,
    sfr_model="Behroozi19",
    model_name="Silva15-m1",
    quantity="intensity",
    line_name="CII158",
    halo_cutoff_mass=1e11,
    nproj=None,
    use_scatter=False,
    halocat_type="input_cat",
    ngrid_x=None,
    ngrid_y=None,
    ngrid_z=None,
    boxsize_x=None,
    boxsize_y=None,
    boxsize_z=None,
    params_fisher=None,
):
    # global lum_line, halomass_cut
    cellsize_x = boxsize_x / ngrid_x
    cellsize_y = boxsize_y / ngrid_y
    cellsize_z = boxsize_z / ngrid_z

    if cellsize_x:
        d_omega_pix = lu.solid_angle(cellsize_x, halo_redshift)
    if cellsize_y:
        d_omega_pix = lu.solid_angle(cellsize_y, halo_redshift)

    d_nu = lu.comoving_size_to_delta_nu(cellsize_z, halo_redshift, line_name=line_name)

    halomass, halo_cm = lu.make_halocat(
        halocat_file, mmin=halo_cutoff_mass, halocat_type=halocat_type
    )

    lum_line = mhalo_to_lline(
        halomass,
        halo_redshift,
        sfr_model=sfr_model,
        model_name=model_name,
        use_scatter=use_scatter,
        line_name=line_name,
        params_fisher=params_fisher,
    )
    # if(quantity=='intensity' or quantity=='Inetnsity' or quantity=='I'):
    #    print("debugging")

    if quantity == "luminosity" or quantity == "Luminosity" or quantity == "L":
        grid_lum = lu.make_grid_rectangular(
            halo_cm,
            weight=lum_line,
            ngrid_x=ngrid_x,
            ngrid_y=ngrid_y,
            ngrid_z=ngrid_z,
            boxsize_x=boxsize_x,
            boxsize_y=boxsize_y,
            boxsize_z=boxsize_z,
        )
        return grid_lum

    if quantity == "intensity":
        grid_lum = lu.make_grid_rectangular(
            halo_cm,
            weight=lum_line,
            ngrid_x=ngrid_x,
            ngrid_y=ngrid_y,
            ngrid_z=ngrid_z,
            boxsize_x=boxsize_x,
            boxsize_y=boxsize_y,
            boxsize_z=boxsize_z,
        )

        D_lum = p.cosmo.D_luminosity(halo_redshift)
        prefac = 4.0204e-2 * small_h**2  # Lsol/Mpc^2/ GHz

        grid_intensity = (
            prefac * (grid_lum / 4.0 / np.pi / D_lum**2) / d_omega_pix / d_nu
        )  # Jy/sr

        return grid_intensity


def make_quantity_rectangular_grid_NO_Z_EVO(
    halocat_file,
    halo_redshift,
    sfr_model="Behroozi19",
    model_name="Silva15-m1",
    quantity="intensity",
    line_name="CII158",
    halo_cutoff_mass=1e11,
    use_scatter=False,
    halocat_type="input_cat",
    ngrid_x=None,
    ngrid_y=None,
    ngrid_z=None,
    boxsize_x=None,
    boxsize_y=None,
    boxsize_z=None,
    nu_obs = None,
    dnu_obs = None,
    params_fisher=None,
):

    if (dnu_obs is not None):
        zem, dz, dchi, d_ngrid = lu.box_freq_to_quantities(nu_obs=nu_obs,
                                                           dnu_obs=dnu_obs,
                                                           boxsize= boxsize_z,
                                                           ngrid= ngrid_z,
                                                           z_start = halo_redshift,
                                                           line_name= line_name)

        Ngrid_new = int(ngrid_z/d_ngrid) if d_ngrid < ngrid_z else 1
        d_ngrid = d_ngrid if d_ngrid< ngrid_z else ngrid_z

    # if dnu_obs is None, then ngrid along z axis will remain unchanged.
    if (dnu_obs is None):
        Ngrid_new = ngrid_z
        d_ngrid = ngrid_z


    Igcal = make_quantity_rectangular_grid(
                halocat_file,
                halo_redshift,
                sfr_model= sfr_model,
                model_name= model_name,
                quantity="intensity",
                line_name=line_name,
                halo_cutoff_mass= halo_cutoff_mass,
                use_scatter= use_scatter,
                halocat_type= halocat_type,
                ngrid_x= ngrid_x,
                ngrid_y= ngrid_y,
                ngrid_z= Ngrid_new,
                boxsize_x= boxsize_x,
                boxsize_y= boxsize_y,
                boxsize_z= boxsize_z,
                params_fisher= params_fisher)

    return Igcal


def make_quantity_rectangular_grid_z_evo(
    halocat_file,
    halo_redshift,
    sfr_model="Behroozi19",
    model_name="Silva15-m1",
    quantity="intensity",
    line_name="CII158",
    halo_cutoff_mass=1e11,
    nproj=None,
    use_scatter=False,
    halocat_type="input_cat",
    ngrid_x=None,
    ngrid_y=None,
    ngrid_z=None,
    boxsize_x=None,
    boxsize_y=None,
    boxsize_z=None,
    nu_obs = None,
    dnu_obs = None,
    params_fisher=None,
):

    if (dnu_obs is not None):
        zem, dz, dchi, d_ngrid = lu.box_freq_to_quantities(nu_obs=nu_obs,
                                                           dnu_obs=dnu_obs,
                                                           boxsize= boxsize_z,
                                                           ngrid= ngrid_z,
                                                           z_start = halo_redshift,
                                                           line_name= line_name)

        Ngrid_new = int(ngrid_z/d_ngrid) if d_ngrid < ngrid_z else 1
        d_ngrid = d_ngrid if d_ngrid< ngrid_z else ngrid_z

    # if dnu_obs is None, then ngrid along z axis will remain unchanged.
    if (dnu_obs is None):
        Ngrid_new = ngrid_z
        d_ngrid = ngrid_z



    # global lum_line, halomass_cut
    cellsize_x = boxsize_x / ngrid_x
    cellsize_y = boxsize_y / ngrid_y
    cellsize_z = boxsize_z / Ngrid_new


    halomass, halo_cm = lu.make_halocat(
        halocat_file, mmin = halo_cutoff_mass, halocat_type=halocat_type
    )


    grid_intensity = np.array([])

    for i in range(Ngrid_new):
        z_start = halo_redshift + (i *dz)

        if cellsize_x:
            d_omega_pix = lu.solid_angle(cellsize_x, z_start)

        if cellsize_y:
            d_omega_pix = lu.solid_angle(cellsize_y, z_start)

        d_nu = lu.comoving_size_to_delta_nu(cellsize_z, z_start, line_name=line_name)


        #print(i)
        zrange =  halo_cm [:,2]
        mask = np.where ((zrange >= i* dchi) & (zrange < (i+1) * dchi))


        x_mask = halo_cm [:,0][mask]
        y_mask  = halo_cm [:,1][mask]
        z_mask  = zrange [mask]

        print(z_mask.min(), z_mask.max())

        halomass_mask = halomass[mask]

        hloc_mask = np.stack((x_mask, y_mask, z_mask), axis=1)

        lum_line = mhalo_to_lline(
                    halomass_mask,
                    z_start,
                    sfr_model=sfr_model,
                    model_name=model_name,
                    use_scatter=use_scatter,
                    line_name=line_name,
                    params_fisher=params_fisher,
                )

        grid_lum = lu.make_grid_rectangular(
            hloc_mask,
            weight=lum_line,
            ngrid_x=ngrid_x,
            ngrid_y=ngrid_y,
            ngrid_z= 1,
            boxsize_x = boxsize_x,
            boxsize_y = boxsize_y,
            boxsize_z = boxsize_z,
        )

        D_lum = p.cosmo.D_luminosity(halo_redshift)
        prefac = 4.0204e-2 * small_h**2  # Lsol/Mpc^2/ GHz

        grid_intensity_cal = (
            prefac * (grid_lum / 4.0 / np.pi / D_lum**2) / d_omega_pix / d_nu
        )  # Jy/sr

        #print("The shape of", np.shape( grid_intensity_cal))
        if i == 0:
            grid_intensity =  grid_intensity_cal
        else:
            grid_intensity = np.dstack([grid_intensity, grid_intensity_cal])
    return grid_intensity

def make_intensity_grid(
    halocat_file,
    halo_redshift,
    sfr_model="Behroozi19",
    model_name="Silva15-m1",
    line_name="CII158",
    halo_cutoff_mass=1e11,
    use_scatter=False,
    halocat_type="input_cat",
    ngrid_x=None,
    ngrid_y=None,
    ngrid_z=None,
    boxsize_x=None,
    boxsize_y=None,
    boxsize_z=None,
    nu_obs = None,
    dnu_obs = None,
    theta_fwhm  = None,
    z_evolution = False,
    params_fisher= []):

    if z_evolution == False:
         Igcal = make_quantity_rectangular_grid_NO_Z_EVO(
                halocat_file,
                halo_redshift,
                sfr_model= sfr_model,
                model_name= model_name,
                quantity="intensity",
                line_name=line_name,
                halo_cutoff_mass= halo_cutoff_mass,
                use_scatter= use_scatter,
                halocat_type= halocat_type,
                ngrid_x= ngrid_x,
                ngrid_y= ngrid_y,
                ngrid_z= ngrid_z,
                boxsize_x= boxsize_x,
                boxsize_y= boxsize_y,
                boxsize_z= boxsize_z,
                nu_obs = nu_obs,
                dnu_obs = dnu_obs,
                params_fisher= params_fisher)

    elif z_evolution == True:
        Igcal = make_quantity_rectangular_grid_z_evo(
                halocat_file,
                halo_redshift,
                sfr_model= sfr_model,
                model_name= model_name,
                quantity="intensity",
                line_name=line_name,
                halo_cutoff_mass= halo_cutoff_mass,
                use_scatter= use_scatter,
                halocat_type= halocat_type,
                ngrid_x= ngrid_x,
                ngrid_y= ngrid_y,
                ngrid_z= ngrid_z,
                boxsize_x= boxsize_x,
                boxsize_y= boxsize_y,
                boxsize_z= boxsize_z,
                nu_obs = nu_obs,
                dnu_obs = dnu_obs,
                params_fisher= params_fisher)


    if theta_fwhm == None:
        return Igcal

    if theta_fwhm:
        Ngrid_z = np.shape(Igcal)[2]
        zem, dz, dchi, d_ngrid = lu.box_freq_to_quantities(nu_obs=nu_obs,
                                                           dnu_obs=dnu_obs,
                                                           boxsize= boxsize_z,
                                                           ngrid= ngrid_z,
                                                           line_name= line_name)

        convolved_grid = []
        theta = lu.convert_beam_unit_to_radian(theta_fwhm, beam_unit= 'arcmin')
        cellsize_x = boxsize_x/ngrid_x
        cellsize_y = boxsize_y/ngrid_y

        for i in range(Ngrid_z):
            z_start = halo_redshift + (i * dz)
            beam_size = lu.angle_to_comoving_size(z_start, theta)
            #beam_std = beam_size / (np.sqrt(8 * np.log10(2.0)))

            beam_std_x = beam_size / (np.sqrt(8 * np.log10(2.0)))/cellsize_x
            beam_std_y = beam_size / (np.sqrt(8 * np.log10(2.0)))/cellsize_y

            gauss_kernel = Gaussian2DKernel(beam_std_x, y_stddev = beam_std_y)
            grid_quantity = Igcal[:,:, i: i+1].reshape(ngrid_x, ngrid_y)
            convolved_grid_cal = convolve(grid_quantity, gauss_kernel)
            convolved_grid.append(convolved_grid_cal)

        Igcal = np.swapaxes(convolved_grid, 0, 2)

        return Igcal


def make_quantity_rectangular_grid_cal(
    halocat_file,
    halo_redshift,
    sfr_model="Behroozi19",
    model_name="Silva15-m1",
    quantity="intensity",
    line_name="CII158",
    halo_cutoff_mass=1e11,
    nproj=None,
    use_scatter=False,
    halocat_type="input_cat",
    ngrid_x=None,
    ngrid_y=None,
    ngrid_z=None,
    boxsize_x=None,
    boxsize_y=None,
    boxsize_z=None,
    nu_obs = None,
    dnu_obs = None,
    redshift_evolution = False,
    params_fisher=None,
):
    # global lum_line, halomass_cut
    cellsize_x = boxsize_x / ngrid_x
    cellsize_y = boxsize_y / ngrid_y
    cellsize_z = boxsize_z / ngrid_z

    if cellsize_x:
        d_omega_pix = lu.solid_angle(cellsize_x, halo_redshift)
    if cellsize_y:
        d_omega_pix = lu.solid_angle(cellsize_y, halo_redshift)

    d_nu = lu.comoving_size_to_delta_nu(cellsize_z, halo_redshift, line_name=line_name)

    halomass, halo_cm = lu.make_halocat(
        halocat_file, mmin = halo_cutoff_mass, halocat_type=halocat_type
    )




    if (dnu_obs is not None):
        zem, dz, dchi, d_ngrid = lu.box_freq_to_quantities(nu_obs=nu_obs,
                                                           dnu_obs=dnu_obs,
                                                           boxsize= boxsize_z,
                                                           ngrid= ngrid_z,
                                                           z_start = halo_redshift,
                                                           line_name= line_name)

        Ngrid_new = int(ngrid_z/d_ngrid) if d_ngrid < ngrid_z else 1
        d_ngrid = d_ngrid if d_ngrid< ngrid_z else ngrid_z

    # if dnu_obs is None, then ngrid along z axis will remain unchanged.
    if (dnu_obs is None):
        Ngrid_new = ngrid_z
        d_ngrid = ngrid_z


    if redshift_evolution == True:

        grid_intensity = np.array([])

        for i in range(Ngrid_new):
            #print(i)
            zrange =  halo_cm [:,2]
            mask = np.where ((zrange >= i* dchi) & (zrange < (i+1) * dchi))
            #print(mask)

            x_mask = halo_cm [:,0][mask]
            y_mask  = halo_cm [:,1][mask]
            z_mask  = zrange [mask]

            halomass_mask = halomass[mask]

            hloc_mask = np.stack((x_mask, y_mask, z_mask), axis=1)

            lum_line = mhalo_to_lline(
                        halomass_mask,
                        halo_redshift + i * dz,
                        sfr_model=sfr_model,
                        model_name=model_name,
                        use_scatter=use_scatter,
                        line_name=line_name,
                        params_fisher=params_fisher,
                    )

            grid_lum = lu.make_grid_rectangular(
                hloc_mask,
                weight=lum_line,
                ngrid_x=ngrid_x,
                ngrid_y=ngrid_y,
                ngrid_z= 1,
                boxsize_x = boxsize_x,
                boxsize_y = boxsize_y,
                boxsize_z = boxsize_z,
            )

            D_lum = p.cosmo.D_luminosity(halo_redshift)
            prefac = 4.0204e-2 * small_h**2  # Lsol/Mpc^2/ GHz

            grid_intensity_cal = (
                prefac * (grid_lum / 4.0 / np.pi / D_lum**2) / d_omega_pix / d_nu
            )  # Jy/sr

            #print("The shape of", np.shape( grid_intensity_cal))
            if i == 0:
                grid_intensity =  grid_intensity_cal
            else:
                grid_intensity = np.dstack([grid_intensity, grid_intensity_cal])
                #grid_intensity = np.stack([grid_intensity, grid_intensity_cal], axis=2)
        #grid_intensity = np.append(grid_intensity, grid_intensity_cal, axis=0)
        #grid_intensity.append(grid_intensity_cal)
        #print("The shape of grid_int", np.shape( grid_intensity))


    if redshift_evolution == False:

        lum_line = mhalo_to_lline(
            halomass,
            halo_redshift,
            sfr_model=sfr_model,
            model_name=model_name,
            use_scatter=use_scatter,
            line_name=line_name,
            params_fisher=params_fisher,
        )
        # if(quantity=='intensity' or quantity=='Inetnsity' or quantity=='I'):
        #    print("debugging")

        grid_lum = lu.make_grid_rectangular(
            halo_cm,
            weight=lum_line,
            ngrid_x=ngrid_x,
            ngrid_y=ngrid_y,
            ngrid_z= Ngrid_new,
            boxsize_x=boxsize_x,
            boxsize_y=boxsize_y,
            boxsize_z=boxsize_z,
        )

        D_lum = p.cosmo.D_luminosity(halo_redshift)
        prefac = 4.0204e-2 * small_h**2  # Lsol/Mpc^2/ GHz

        grid_intensity = (
            prefac * (grid_lum / 4.0 / np.pi / D_lum**2) / d_omega_pix / d_nu
        )  # Jy/sr

    return grid_intensity




def plot_beam_convolution(
    convolved_grid,
    ngrid,
    boxsize,
    halo_redshift,
    plot_unit="mpc",
    quantity="intensity",
    cmap="gist_heat",
    tick_num=5,
    vmin=None,
    vmax=None,
    title="",
    plot_scale="log",
):

    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)

    radian_to_minute = (180.0 * 60.0) / np.pi
    radian_to_degree = (180.0) / np.pi

    if vmin == None:
        vmin = 0.1
    if vmax == None:
        vmax = convolved_grid.max()

    if plot_scale == "log":
        convolved_grid[
            convolved_grid <= 0
        ] = 1e-20  # Fill zero or negative values with a very small number so that log(0) does not exist.
        res = ax.imshow(
            np.log10(convolved_grid),
            cmap=cmap,
            interpolation="gaussian",
            origin="lower",
            rasterized=True,
            alpha=0.9,
            vmin=np.log10(vmin),
            vmax=np.log10(vmax),
        )
        plt.title(title)
        if quantity == "intensity":
            colorbar_label = r"$\mathrm{log}\,I_{\rm line}$"
        if quantity == "luminosity":
            colorbar_label = r"$\mathrm{log}\,L_{\rm line}$"

    if plot_scale == "lin":
        res = ax.imshow(
            convolved_grid,
            cmap=cmap,
            interpolation="gaussian",
            origin="lower",
            rasterized=True,
            alpha=0.9,
            vmin=vmin,
            vmax=vmax,
        )
        plt.title(title)
        colorbar_label = r"$L_{\rm line}$"

        if quantity == "intensity":
            colorbar_label = r"$I_{\rm line}$"
        if quantity == "luminosity":
            colorbar_label = r"$L_{\rm line}$"

    if plot_unit == "degree":
        x_tick = (
            lu.comoving_boxsize_to_angle(halo_redshift, boxsize)
        ) * radian_to_degree
        cell_size = x_tick / ngrid
        ticks = np.linspace(0, x_tick, num=tick_num)
        labels = [str("{:.1f}".format(xx)) for xx in ticks]
        locs = [xx / cell_size for xx in ticks]
        plt.xlabel(r"$\Theta\,(\mathrm{degree})$")
        plt.ylabel(r"$\Theta\,(\mathrm{degree})$")

    if plot_unit == "minute":
        x_tick = (
            lu.comoving_boxsize_to_angle(halo_redshift, boxsize)
        ) * radian_to_minute
        cell_size = x_tick / ngrid
        ticks = np.linspace(0, x_tick, num=tick_num)
        labels = [str("{:.1f}".format(xx)) for xx in ticks]
        locs = [xx / cell_size for xx in ticks]
        plt.xlabel(r"$\Theta\,(\mathrm{arc-min})$")
        plt.ylabel(r"$\Theta\,(\mathrm{arc-min})$")

    if plot_unit == "mpc":
        x_tick = boxsize
        cell_size = boxsize / ngrid
        ticks = np.linspace(0, x_tick, num=tick_num)
        labels = [str("{:.1f}".format(xx)) for xx in ticks]
        locs = [xx / cell_size for xx in ticks]
        plt.xlabel(r"$X\,(\mathrm{Mpc})$")
        plt.ylabel(r"$Y\,(\mathrm{Mpc})$")

    plt.xticks(locs, labels)
    plt.yticks(locs, labels)

    # title = '$z={:g}$'.format(halo_redshift)
    # plt.title(title, fontsize=18)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = plt.colorbar(res, cax=cax)
    cb.set_label(colorbar_label, labelpad=20)
    cb.solids.set_edgecolor("face")
    cb.ax.tick_params("both", which="major", length=3, width=1, direction="out")

    plt.tight_layout()
    plt.savefig("luminsoty_beam.png")


def plot_slice_zoomin(
    convolved_grid,
    ngrid,
    boxsize,
    halo_redshift,
    origin=[0, 0],
    cmap="hot",
    slice_size=5,
    plot_unit="mpc",
    quantity="intensity",
    tick_num=5,
    vmin=None,
    vmax=None,
    plot_scale="log",
):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    radian_to_minute = (180.0 * 60.0) / np.pi
    radian_to_degree = (180.0) / np.pi
    cellsize_mpc = boxsize / ngrid

    if vmin == None:
        vmin = 0.1
    if vmax == None:
        vmax = convolved_grid.max()

    slice_szie_to_ngrid = int(slice_size / cellsize_mpc)

    if origin == [0, 0]:
        x_grid = slice_szie_to_ngrid
        y_grid = x_grid
        convolved_grid = convolved_grid[0 : x_grid + 1, 0 : y_grid + 1]

    else:
        x_grid_origin = int(origin[0] / cellsize_mpc)
        y_grid_origin = int(origin[1] / cellsize_mpc)

        x_grid = x_grid_origin + slice_szie_to_ngrid
        y_grid = y_grid_origin + slice_szie_to_ngrid
        convolved_grid = convolved_grid[
            x_grid_origin : x_grid + 1, y_grid_origin : y_grid + 1
        ]

    if plot_scale == "log":
        convolved_grid[
            convolved_grid <= 0
        ] = 1e-20  # Fill zero or negative values with a very small number so that log(0) does not exist.
        res = ax.imshow(
            np.log10(convolved_grid),
            cmap=cmap,
            interpolation="gaussian",
            origin="lower",
            rasterized=True,
            alpha=0.9,
            vmin=np.log10(vmin),
            vmax=np.log10(vmax),
        )
        if quantity == "intensity":
            colorbar_label = r"$\mathrm{log}\,I_{\rm line}$"
        if quantity == "luminosity":
            colorbar_label = r"$\mathrm{log}\,L_{\rm line}$"

    if plot_scale == "lin":
        res = ax.imshow(
            convolved_grid,
            cmap=cmap,
            interpolation="gaussian",
            origin="lower",
            rasterized=True,
            alpha=0.9,
            vmin=vmin,
            vmax=vmax,
        )
        colorbar_label = r"$L_{\rm line}$"

        if quantity == "intensity":
            colorbar_label = r"$I_{\rm line}$"
        if quantity == "luminosity":
            colorbar_label = r"$L_{\rm line}$"

    if plot_unit == "degree":
        x_tick = (
            lu.comoving_boxsize_to_angle(halo_redshift, boxsize)
        ) * radian_to_degree
        cell_size = x_tick / ngrid
        ticks = np.linspace(0, x_tick, num=tick_num)
        labels = [str("{:.1f}".format(xx)) for xx in ticks]
        locs = [xx / cell_size for xx in ticks]
        plt.xlabel(r"$\Theta\,(\mathrm{degree})$")
        plt.ylabel(r"$\Theta\,(\mathrm{degree})$")

    if plot_unit == "minute":
        x_tick = (
            lu.comoving_boxsize_to_angle(halo_redshift, boxsize)
        ) * radian_to_minute
        cell_size = x_tick / ngrid
        ticks = np.linspace(0, x_tick, num=tick_num)
        labels = [str("{:.1f}".format(xx)) for xx in ticks]
        locs = [xx / cell_size for xx in ticks]
        plt.xlabel(r"$\Theta\,(\mathrm{arc-min})$")
        plt.ylabel(r"$\Theta\,(\mathrm{arc-min})$")

    if plot_unit == "mpc":
        x_tick = slice_size
        cell_size = boxsize / ngrid
        ticks = np.linspace(0, x_tick, num=tick_num)
        labels = [str("{:.1f}".format(xx)) for xx in ticks]
        locs = [xx / cell_size for xx in ticks]
        plt.xlabel(r"$X\,(\mathrm{Mpc})$")
        plt.ylabel(r"$Y\,(\mathrm{Mpc})$")

    plt.xticks(locs, labels)
    plt.yticks(locs, labels)

    # title = '$z={:g}$'.format(halo_redshift)
    # plt.title(title, fontsize=18)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = plt.colorbar(res, cax=cax)
    cb.set_label(colorbar_label, labelpad=20)
    cb.solids.set_edgecolor("face")
    cb.ax.tick_params("both", which="major", length=3, width=1, direction="out")

    plt.tight_layout()
    plt.savefig("luminsoty_beam.png")


def calc_intensity_3d(
    boxsize,
    ngrid,
    halocat_file,
    halo_redshift,
    sfr_model="Behroozi19",
    line_name="CII158",
    halo_cutoff_mass=1e11,
    use_scatter=False,
    halocat_type="input_cat",
    intensity_unit="jy/sr",
    params_fisher=None,
):
    """
    Calculate luminosity for input parameters
    """

    # low_mass_log=0.0
    cellsize = boxsize / ngrid

    V_cell = (cellsize) ** 3

    halomass, halo_cm = lu.make_halocat(
        halocat_file, mmin=halo_cutoff_mass, halocat_type=halocat_type, boxsize=boxsize
    )

    lcp = mhalo_to_lline(
        halomass,
        halo_redshift,
        sfr_model=sfr_model,
        line_name=line_name,
        use_scatter=use_scatter,
        params_fisher=params_fisher,
    )

    grid_lum = lu.grid(halo_cm, lcp, boxsize, ngrid, ndim=3)

    prefac = p.c_in_m / (
        4
        * np.pi
        * p.nu_rest(line_name=line_name)
        * p.Ghz_to_hz
        * p.cosmo.H_z(halo_redshift)
    )

    if intensity_unit == "jy" or intensity_unit == "Jy" or intensity_unit == "JY":
        grid_intensity = (
            prefac * (grid_lum * p.Lsun / (V_cell * p.mpc_to_m**3)) / p.jy_unit
        )  # transformed to jansky unit

    if (
        intensity_unit == "jy/sr"
        or intensity_unit == "Jy/sr"
        or intensity_unit == "JY/sr"
    ):
        grid_intensity = (
            prefac
            * (grid_lum * p.Lsun / (V_cell * p.mpc_to_m**3))
            / p.jy_unit
            / (4 * np.pi)
        )  # JY/sr unit

    return grid_intensity


def calc_intensity_2d(
    boxsize,
    ngrid,
    halocat_file,
    halo_redshift,
    sfr_model="Behroozi19",
    model_name="Silva15-m1",
    line_name="CII158",
    halo_cutoff_mass=1e11,
    use_scatter=False,
    halocat_type="input_cat",
    intensity_unit="jy/sr",
    params_fisher=None,
):
    """
    Calculate luminosity for input parameters
    """

    cellsize = boxsize / ngrid

    V_cell = (cellsize) ** 3

    halomass, halo_cm = lu.make_halocat(
        halocat_file, mmin=halo_cutoff_mass, halocat_type=halocat_type, boxsize=boxsize
    )

    nhalo = len(halomass)
    x_halos = halo_cm[range(0, nhalo * 3, 3)]
    y_halos = halo_cm[range(1, nhalo * 3, 3)]
    z_halos = halo_cm[range(2, nhalo * 3, 3)]

    print("Minimum halo mass:", halomass.min())
    print("Maximum halo mass:", halomass.max())

    halo_cm = np.concatenate([x_halos, y_halos, z_halos])

    lcp = mhalo_to_lline(
        halomass,
        halo_redshift,
        sfr_model=sfr_model,
        model_name=model_name,
        line_name=line_name,
        use_scatter=use_scatter,
        params_fisher=params_fisher,
    )

    grid_lum = lu.grid(halo_cm, lcp, boxsize, ngrid, ndim=3)

    # print("shape of grid_lum", np.shape(grid_lum))

    prefac = p.c_in_m / (
        4
        * np.pi
        * p.nu_rest(line_name=line_name)
        * p.Ghz_to_hz
        * p.cosmo.H_z(halo_redshift)
    )

    # print("shape of prefac", np.shape(prefac))

    if intensity_unit == "jy" or intensity_unit == "Jy" or intensity_unit == "JY":
        grid_intensity = (
            prefac * (grid_lum * p.Lsun / (V_cell * p.mpc_to_m**3)) / p.jy_unit
        )  # transformed to jansky unit

    if (
        intensity_unit == "jy/sr"
        or intensity_unit == "Jy/sr"
        or intensity_unit == "JY/sr"
    ):
        grid_intensity = (
            prefac
            * (grid_lum * p.Lsun / (V_cell * p.mpc_to_m**3))
            / p.jy_unit
            / (4 * np.pi)
        )  # JY/sr unit

    return grid_intensity


def intensity_power_spectra(
    boxsize,
    ngrid,
    halocat_file,
    halo_redshift,
    line_name="CII158",
    project_length=None,
    halo_cutoff_mass=1e11,
    use_scatter=False,
    halocat_file_type="dat",
    intensity_unit="jy/sr",
    remove_shotnoise=False,
    volume_normalization=False,
    params_fisher=None,
):

    I_grid = calc_intensity_3d(
        boxsize,
        ngrid,
        halocat_file,
        halo_redshift,
        line_name=line_name,
        halo_cutoff_mass=halo_cutoff_mass,
        use_scatter=use_scatter,
        halocat_file_type=halocat_file_type,
        intensity_unit=intensity_unit,
        params_fisher=params_fisher,
    )

    k, pk = lu.powerspectra_2d(
        I_grid,
        boxsize,
        ngrid,
        project_length=project_length,
        volume_normalization=volume_normalization,
        remove_shotnoise=remove_shotnoise,
    )

    return k, pk






def get_beam_cov_3d(grid_quantity, halo_redshift,
    theta_fwhm,
    beam_unit,
    line_name ="CII158",
    boxsize_x=None,
    boxsize_y=None,
    boxsize_z=None,
    ngrid_x=None,
    ngrid_y=None,
    ngrid_z=None,
    nu_obs = None,
    dnu_obs = None):

    zem, dz, dchi, d_ngrid = lu.box_freq_to_quantities(nu_obs=nu_obs, dnu_obs=dnu_obs, boxsize= boxsize_z, ngrid= ngrid_z, line_name= line_name)
    Ngrid_new = int(ngrid_z/d_ngrid) if d_ngrid < ngrid_z else 1
    d_ngrid = d_ngrid if d_ngrid< ngrid_z else ngrid_z

    #print ("The reduced Ngrid is", Ngrid_new, d_ngrid)


    theta = lu.convert_beam_unit_to_radian(theta_fwhm, beam_unit=beam_unit)
    beam_size = lu.angle_to_comoving_size(halo_redshift, theta)

    cellsize_x = boxsize_x / ngrid_x
    cellsize_y = boxsize_y / ngrid_y

    beam_std_x = beam_size / (np.sqrt(8 * np.log10(2.0)))/cellsize_x
    beam_std_y = beam_size / (np.sqrt(8 * np.log10(2.0)))/cellsize_y

    gauss_kernel = Gaussian2DKernel(beam_std_x, y_stddev = beam_std_y)


    convolved_grid = []

    for i in range(Ngrid_new):
        grid_start = i*d_ngrid
        grid_end = (i+1)*d_ngrid
        #print(grid_start, grid_end)
        grid_quantity1 = np.mean(grid_quantity[:,:, grid_start: grid_end], axis=2)
        convolved_grid_cal = convolve(grid_quantity1, gauss_kernel)
        convolved_grid.append(convolved_grid_cal)

    return np.swapaxes(convolved_grid, 0, 2)
