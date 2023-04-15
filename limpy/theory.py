#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:55:42 2022

@author: anirbanroy
"""

import numpy as np
from scipy.integrate import simps

# from scipy.interpolate import interp1d
from scipy.special import erf

import limpy.lines as ll
import limpy.params as p

################################################################################
## initialization
################################################################################

small_h = p.cosmo.h


################################################################################
##      HOD model and Halo mass function
################################################################################


def N_cen(M, M_min=10**8, sigma_logm=0.15):
    """
    # Note, we consider M_min = 10** 8 M_sun, whereas in original paper it was
    10**11.57.
    """
    # M = M/small_h
    res = 0.5 * (1 + erf((np.log10(M) - np.log10(M_min)) / sigma_logm))
    return res


def N_sat(M, M_cut=10**12.23, M1=10**12.75, alpha_g=0.99):
    # M = M/small_h
    if np.isscalar(M):
        if M <= M_cut:
            res = 0.0
        else:
            res = N_cen(M) * ((M - M_cut) / M1) ** alpha_g

    else:
        below_M_cut = np.where(M < M_cut)[0]
        above_M_cut = np.where(M >= M_cut)[0]

        res = np.zeros(len(M))
        res[below_M_cut] = 0.0
        res[above_M_cut] = (
            N_cen(M[above_M_cut]) * ((M[above_M_cut] - M_cut) / M1) ** alpha_g
        )

    return res


def N_M_hod(Mh):
    return (N_sat(Mh) + 1) * N_cen(Mh)


def hmf(z, Mmin=p.Mmin, Mmax=p.Mmax, HOD_model=False):
    Mass_bin, dndm = p.cosmo.hmf_setup(z, Mmin, Mmax)

    if HOD_model:
        return Mass_bin, dndm * N_M_hod(Mass_bin)

    else:
        return Mass_bin, dndm


def I_line(
    z, line_name="CII158", model_name="Silva15-m1", sfr_model="Silva15", HOD_model=False
):
    # mass_range=np.logspace(np.log10(Mmin), np.log10(Mmax),num=500)
    mass_bin, dndlnM = hmf(z, HOD_model=HOD_model)

    L_line = ll.mhalo_to_lline(
        mass_bin, z, line_name=line_name, sfr_model=sfr_model, model_name=model_name
    )  # L_sun unit

    factor = (p.c_in_mpc) / (
        4 * np.pi * p.nu_rest(line_name=line_name) * p.cosmo.H_z(z)
    )

    conversion_fac = 4.0204e-2  # jy/sr

    integrand = factor * dndlnM * L_line

    integration = simps(integrand, np.log(mass_bin))

    return integration * conversion_fac




def P_shot_gong(
    z, line_name="CII158", sfr_model="Silva15", model_name="Silva15-m1", HOD_model=False
):
    # mass_range=np.logspace(np.log10(Mmin), np.log10(Mmax),num=500)
    mass_bin, dndlnM = hmf(z, HOD_model=HOD_model)

    L_line = ll.mhalo_to_lline(
        mass_bin, z, line_name=line_name, sfr_model=sfr_model, model_name=model_name
    )

    integrand_numerator = dndlnM * L_line**2
    int_numerator = simps(integrand_numerator, np.log(mass_bin))


    factor = (p.c_in_mpc) / ( 4 * np.pi * p.nu_rest(line_name=line_name) * p.cosmo.H_z(z))

    conversion_fac = 4.0204e-2 # jy/sr

    return int_numerator * factor**2 * conversion_fac**2


def T_line(
    z,
    line_name="CII158",
    sfr_model="Silva15",
    model_name="Silva15-m1",
    fduty=1.0,
    HOD_model=False,
):
    Intensity = I_line(
        z,
        line_name=line_name,
        model_name=model_name,
        sfr_model=sfr_model,
        HOD_model=HOD_model,
    )
    nu_obs = p.nu_rest(line_name=line_name) / (1 + z)
    T_line = p.c_in_mpc**2 * Intensity / (2 * p.k_b * nu_obs**2)

    return T_line  # in muK


def P_shot(
    z, line_name="CII158", sfr_model="Silva15", model_name="Silva15-m1", HOD_model=False
):
    # mass_range=np.logspace(np.log10(Mmin), np.log10(Mmax),num=500)
    mass_bin, dndlnM = hmf(z, HOD_model=HOD_model)

    L_line = ll.mhalo_to_lline(
        mass_bin, z, line_name=line_name, sfr_model=sfr_model, model_name=model_name
    )

    integrand_numerator = dndlnM * L_line**2
    integrand_denominator = dndlnM * L_line
    int_numerator = simps(integrand_numerator, np.log(mass_bin))
    int_denominator = simps(integrand_denominator, np.log(mass_bin))

    return int_numerator / int_denominator**2



def b_line(
    z, line_name="CII158", sfr_model="Silva15", model_name="Silva15-m1", HOD_model=False
):
    mass_bin, dndlnM = hmf(z, HOD_model=HOD_model)
    L_line = ll.mhalo_to_lline(
        mass_bin, z, line_name=line_name, sfr_model=sfr_model, model_name=model_name
    )

    integrand_numerator = dndlnM * L_line * p.cosmo.bias_dm(mass_bin, z)
    integrand_denominator = dndlnM * L_line
    int_numerator = simps(integrand_numerator, np.log(mass_bin))
    int_denominator = simps(integrand_denominator, np.log(mass_bin))

    return int_numerator / int_denominator


def Pk_line(
    k,
    z,
    fduty=1.0,
    line_name="CII158",
    label="total",
    sfr_model="Silva15",
    model_name="Silva15-m1",
    pk_unit="intensity",
    HOD_model=False,
):
    if pk_unit == "intensity":
        I_nu_square = (
            I_line(
                z,
                line_name=line_name,
                model_name=model_name,
                sfr_model=sfr_model,
                HOD_model=HOD_model,
            )
            ** 2
        )
        pk_lin = p.cosmo.pk_camb(k, z)

        if label == "total":
            res = I_nu_square * b_line(
                z, line_name=line_name, model_name=model_name, HOD_model=HOD_model
            ) ** 2 * pk_lin + I_nu_square * P_shot(
                z, line_name=line_name, model_name=model_name, HOD_model=HOD_model
            )

        if label == "clustering":
            res = (
                I_nu_square
                * b_line(
                    z, line_name=line_name, model_name=model_name, HOD_model=HOD_model
                )
                ** 2
                * pk_lin
            )

        if label == "shot":
            res = I_nu_square * P_shot(
                z, line_name=line_name, model_name=model_name, HOD_model=HOD_model
            )

        return res

    if pk_unit == "temperature" or pk_unit == "muk":
        T_line_square = (
            T_line(
                z,
                line_name=line_name,
                model_name=model_name,
                sfr_model=sfr_model,
                fduty=fduty,
            )
            ** 2
        )
        pk_lin = p.cosmo.pk_camb(k, z)
        if label == "total":
            res = T_line_square * (
                b_line(
                    z, line_name=line_name, model_name=model_name, HOD_model=HOD_model
                )
                ** 2
                * pk_lin
                + P_shot(
                    z, line_name=line_name, model_name=model_name, HOD_model=HOD_model
                )
            )

        if label == "clustering":
            res = T_line_square * (
                b_line(
                    z,
                    line_name=line_name,
                    sfr_model=sfr_model,
                    model_name=model_name,
                    HOD_model=HOD_model,
                )
                ** 2
                * pk_lin
            )

        if label == "shot":
            res = T_line_square * (
                P_shot(
                    z,
                    line_name=line_name,
                    sfr_model=sfr_model,
                    model_name=model_name,
                    HOD_model=HOD_model,
                )
            )
            res = res

        return res


def window_gauss(z, z_mean, deltaz):
    p = (1.0 / np.sqrt(2 * np.pi * deltaz)) * np.exp(
        -((z - z_mean) ** 2) / 2.0 / deltaz**2
    )
    return p


def Cl_line(
    ell,
    z,
    deltaz,
    fduty=1.0,
    line_name="CII158",
    label="total",
    sfr_model="Silva15",
    model_name="Silva15-m1",
    pk_unit="temperature",
):

    chi =  p.cosmo.D_co(z)

    kp = ell / chi

    pk = Pk_line(
        kp,
        z,
        fduty=fduty,
        line_name=line_name,
        sfr_model=sfr_model,
        model_name=model_name,
        label=label,
        pk_unit=pk_unit,
    )
    # pk=dk*2*np.pi**2/kp**3

    zint = np.logspace(np.log10(z - deltaz), np.log10(z + deltaz), num=50)

    integrand = (
        1.0
        / (p.c_in_mpc)
        * (
            window_gauss(zint, z, deltaz)
            * p.cosmo.H_z(z)
            * pk[:, np.newaxis]
            / chi ** 2
        )
    )
    res = simps(integrand, zint, axis=1)
    return res
