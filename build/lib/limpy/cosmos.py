#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:57:30 2020

@author: anirbanroy
"""

import numpy as np

import params as p
import scipy.integrate as si
import camb
from camb import model, initialpower
from camb import get_matter_power_interpolator
class cosmo():
    
    def __init__(self):
        self.h=p.cosmo_params['h']
        self.ol=p.cosmo_params['omega_lambda']
        self.ob=p.cosmo_params['omgega_bh2']/self.h**2
        self.om=p.cosmo_params['omega_mh2']/self.h**2
        self.ocdm=p.cosmo_params['omgega_ch2']/self.h**2
        self.ns=p.cosmo_params['ns']
        self.H0=100*self.h # Km/S/Mpc
           
        if p.cosmo_params['omega_k'] is not None:
            self.ok=p.cosmo_params['omega_k']
        else:
            self.ok=1-(p.cosmo_params['omega_mh2']/self.h**2+ self.ol)
            
       
        self.G_const=p.default_constants['G_const']
        self.c_in_m=p.default_constants['c_in_m']
        self.mpc_to_m=p.default_constants['mpc_to_m']
        self.km_to_m=p.default_constants['km_to_m']
            
        
        
        
    def E_z(self,z):
        return np.sqrt(self.om*(1+z)**3+self.ok*(1+z)**2+self.ol)
    
    def H_z(self, z):
        '''
        Hubble constant at redshift z. 
        unit: s^{-1}
        '''
        return 100*self.h*np.sqrt(self.om*(1+z)**3+self.ok*(1+z)**2+self.ol)\
                *self.km_to_m/self.mpc_to_m

    
    def D_co_unvec(self,z):
        '''
        Comoving distance transverse.
        '''
        omega_k_abs=abs(self.ok)
        D_H=self.c_in_m/(self.H0*self.km_to_m/self.mpc_to_m)
 
        D_c_int=lambda z: D_H/self.E_z(z)
        D_c=si.quad(D_c_int,0,z,limit=1000)[0]
        
        
        if (self.ok==0):
            return D_c
        elif (self.ok<0):
            return D_H/np.sqrt(omega_k_abs)*np.sin(np.sqrt(omega_k_abs)*D_c/D_H)
        elif (self.ok>0):
            return D_H*np.sinh(np.sqrt(self.ok)*D_c/D_H)/np.sqrt(self.ok)


    def D_co(self,z):
        if np.isscalar(z):
            return self.D_co_unvec(z)
        else:
            result_array=np.zeros(len(z))
            for i in range(len(z)):
               result_array[i]=self.D_co_unvec(z[i])
            
            return result_array
    
    def D_angular(self,z):
        '''
        Angular diameter distance
        '''
        #res= self.D_co(z)
        if np.isscalar(z):
            return self.D_co_unvec(z)/(1+z)
        else:
            return [self.D_co(zin)/(1+zin) for zin in z]
    
    
    def D_luminosity(self,z):
        '''
        Angular diameter distance
        '''
        
        return [self.D_angular(z[i])/(1+z[i])**2 for i in(len[z])]
    
    def pk_camb(self,k,z, kmax=10.0):       
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=100*self.h, ombh2=self.ob*self.h**2, omch2=self.ocdm*self.h**2)
        pars.InitPower.set_params(ns=self.ns)

        pars.set_matter_power(redshifts=[z], kmax=kmax)
        
        PK = get_matter_power_interpolator(pars);
    
        return PK.P(z,k)
        
        
        '''
        if(spectra=='nonlinear'):
            #Non-Linear spectra (Halofit)
            pars.NonLinear = model.NonLinear_both
            results.calc_power_spectra(pars)
            kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
            return kh_nonlin, z_nonlin, pk_nonlin
        '''
    
# To test the code with Cosmolopy   


cosmop = {'omega_M_0' : 0.311, 
'omega_lambda_0' : 0.6899, 
'omega_b_0' : 0.048974, 
'omega_n_0' : 0.0,
'omega_k_0' : -0.05,
'N_nu' : 0,
'h' : 0.6766,
'n' : 1.0,
'sigma_8' : 0.9
} 