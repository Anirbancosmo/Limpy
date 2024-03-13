#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:55:42 2022

@author: anirbanroy
"""

import numpy as np
from scipy.integrate import simpson as simps
from scipy.special import erf
import limpy.lines as ll
import limpy.cosmos as cosmos
import limpy.inputs as inp


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
    