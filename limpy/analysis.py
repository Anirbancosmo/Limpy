#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:18:42 2022

@author: anirbanroy
"""
import numpy as np
import limpy.theory as lt
import limpy.utils as lu



def SNR_pk(k, pk_signal, pk_noise, Vsurv, deltak=None, bin_num=10):

    if np.isscalar(pk_noise) == True:
        pk_noise = pk_noise * np.ones(len(k))
    else:
        pk_noise = pk_noise

    if deltak == None:
        knew = np.logspace(np.log10(k.min()), np.log10(k.max()), num=bin_num)
        deltak = np.diff(knew)
        k_cen = (knew[1:] + knew[:-1]) / 2

    else:
        knew = np.arange(k.min(), k.max(), deltak)
        deltak = np.diff(knew)
        k_cen = (knew[1:] + knew[:-1]) / 2

    Nmodes = 2 * np.pi * k_cen**2 * deltak * Vsurv / (2 * np.pi) ** 3

    pksignal_int = np.interp(k_cen, k, pk_signal)
    pknoise_int = np.interp(k_cen, k, pk_noise)

    var = (pksignal_int + pknoise_int) ** 2 / Nmodes

    return k_cen, pksignal_int / np.sqrt(var)




def SNR_pk_cross(k, pk_signal_xy, pk_signal_x, pk_noise_x, pk_signal_y, pk_noise_y, Vsurv, deltak=None, bin_num=10):

    if np.isscalar(pk_noise_x) == True:
        pk_noise_x = pk_noise_x * np.ones(len(k))
    else:
        pk_noise_x = pk_noise_y

    if np.isscalar(pk_noise_y) == True:
        pk_noise_y = pk_noise_y * np.ones(len(k))
    else:
        pk_noise_y = pk_noise_y


    if deltak == None:
        knew = np.logspace(np.log10(k.min()), np.log10(k.max()), num=bin_num)
        deltak = np.diff(knew)
        k_cen = (knew[1:] + knew[:-1]) / 2

    else:
        knew = np.arange(k.min(), k.max(), deltak)
        deltak = np.diff(knew)
        k_cen = (knew[1:] + knew[:-1]) / 2

    Nmodes = 2 * np.pi * k_cen**2 * deltak * Vsurv / (2 * np.pi) ** 3

    pksignal_int_xy = np.interp(k_cen, k, pk_signal_xy)

    pksignal_int_x = np.interp(k_cen, k, pk_signal_x)
    pknoise_int_x = np.interp(k_cen, k, pk_noise_x)

    pksignal_int_y = np.interp(k_cen, k, pk_signal_y)
    pknoise_int_y = np.interp(k_cen, k, pk_noise_y)

    var = ((pksignal_int_x + pknoise_int_x) * (pksignal_int_y + pknoise_int_y) )/ Nmodes

    return k_cen, pksignal_int_xy / np.sqrt(var)


def V_field_CII(z, dnu_obs=2.8, area= 16):
    "Survey field for CII lines" # given by Gong et al. 2012

    if dnu_obs is None:
        res= 3.7e7 * np.sqrt((1+z)/8)* (area/ 16) # assume B_nu=20

    if dnu_obs is not None:
        res= 3.7e7 * np.sqrt((1+z)/8)* (area/ 16) * (dnu_obs/ 20)

    return res


def pk_error(k, pk_signal, pk_noise, Vsurv, noise_prop = "SN", bin_num = 10, binning_scheme = "log",
             binning_method = "average", kmin = None, kmax = None):

    if kmin == None:
        kmin = k.min()
    else:
        kmin = kmin


    if kmax == None:
        kmax = k.max()
    else:
        kmax = kmax


    if binning_scheme == "log":
        knew = np.logspace(np.log10(kmin), np.log10(kmax), num=(bin_num+1))

    if binning_scheme == "linear":
        knew = np.linspace(kmin, kmax, num=(bin_num+1))

    if np.isscalar(pk_noise) == True:
        pk_noise = pk_noise * np.ones(len(k))
    else:
        pk_noise = pk_noise

    deltak = np.diff(knew)
    k_cen = (knew[1:] + knew[:-1]) / 2

    if binning_method == "interpolation":
        Nmodes = 2 * np.pi * k_cen**2 * deltak * Vsurv / (2 * np.pi) ** 3

        pksignal_int = np.interp(k_cen, k, pk_signal)

        pknoise_int = np.interp(k_cen, k, pk_noise)

        var = (pksignal_int + pknoise_int) ** 2 / Nmodes

    if binning_method == "average":
        Nmodes = 2 * np.pi * k_cen**2 * deltak * Vsurv / (2 * np.pi) ** 3

        pksignal_int = np.zeros(bin_num)
        pknoise_int = np.zeros(bin_num)

        for i in range(bin_num):
            kstart = kmin
            kend = kstart + deltak[i]
            kmin = kmin + deltak[i]

            k_indices = np.where(np.logical_and(k >= kstart,  k <= kend))
            pksignal_int[i] = np.mean(pk_signal[k_indices] )

            pknoise_int[i] = np.mean( pk_noise[k_indices] )

    if noise_prop == "ON":
        var = (pksignal_int + pknoise_int) ** 2 / Nmodes

    if (noise_prop == "SN" or noise_prop == "NS"):
        var = (pknoise_int) ** 2 / Nmodes

    return k_cen, pksignal_int, np.sqrt(var)
