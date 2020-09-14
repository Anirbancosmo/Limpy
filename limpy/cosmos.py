#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:57:30 2020

@author: anirbanroy
"""

import numpy as np

import inputs as inp
import scipy.integrate as si
import camb
from camb import get_matter_power_interpolator
class cosmo():
    
    def __init__(self):
        self.h=inp.cosmo_params['h']
        self.ol=inp.cosmo_params['omega_lambda']
        self.ob=inp.cosmo_params['omgega_bh2']/self.h**2
        self.om=inp.cosmo_params['omega_mh2']/self.h**2
        self.ocdm=inp.cosmo_params['omgega_ch2']/self.h**2
        self.ns=inp.cosmo_params['ns']
        self.H0=100*self.h # Km/S/Mpc
           
        if inp.cosmo_params['omega_k'] is not None:
            self.ok=inp.cosmo_params['omega_k']
        else:
            self.ok=1-(inp.cosmo_params['omega_mh2']/self.h**2+ self.ol)
       
        self.G_const=inp.default_constants['G_const']
        self.c_in_m=inp.default_constants['c_in_m']
        self.mpc_to_m=inp.default_constants['mpc_to_m']
        self.km_to_m=inp.default_constants['km_to_m']
            
        
        
        
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