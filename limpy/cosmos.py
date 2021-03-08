#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:57:30 2020

@author: anirbanroy
"""

import numpy as np

import limpy.inputs as inp
import scipy.integrate as si
import camb
from camb import get_matter_power_interpolator

from colossus.lss import mass_function
from colossus.lss import bias

from colossus.cosmology import cosmology as col_cosmology

class cosmo():
    
    def __init__(self):
        self.h=inp.cosmo_params['h']
        self.ol=inp.cosmo_params['omega_lambda']
        self.ob=inp.cosmo_params['omgega_bh2']/self.h**2
        self.om=inp.cosmo_params['omega_mh2']/self.h**2
        self.ocdm=inp.cosmo_params['omgega_ch2']/self.h**2
        self.ns=inp.cosmo_params['ns']
        self.sigma8=inp.cosmo_params['sigma8']
        
        self.H0=100*self.h # Km/S/Mpc
           
        if inp.cosmo_params['omega_k'] is not None:
            self.ok=inp.cosmo_params['omega_k']
        else:
            self.ok=1-(inp.cosmo_params['omega_mh2']/self.h**2+ self.ol)
       
        self.G_const=inp.default_constants['G_const']
        self.c_in_m=inp.default_constants['c_in_m']
        self.mpc_to_m=inp.default_constants['mpc_to_m']
        self.km_to_m=inp.default_constants['km_to_m']
        
        self.halo_model=inp.astro_params['halo_model']
        
        
        
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
    
    
    def pk_lin_camb(self,k,z, kmax=10.0):
        
        
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.H0, ombh2=self.ob*self.h**2, omch2=self.ocdm*self.h**2)
        pars.InitPower.set_params(ns=self.ns)
        
        pars.set_matter_power(redshifts=[z], kmax=2.0)
        pk = get_matter_power_interpolator(pars)
        
        
        return pk
    
    
    
    def set_cosmo_colossus(self):
        params = {'flat': True, 'H0': self.H0, 'Om0': self.om, 'Ob0': self.ob, 'sigma8': self.sigma8, 'ns': self.ns}
        col_cosmology.addCosmology('myCosmo', params)
        cosmo_col=col_cosmology.setCosmology('myCosmo')
        
        print(cosmo_col)



    def hmf(self, z,  Mmin, Mmax, mdef = 'fof', q_out = 'dndlnM', print_cosmo=False):
        
        params = {'flat': True, 'H0': self.H0, 'Om0': self.om, 'Ob0': self.ob, 'sigma8': self.sigma8, 'ns': self.ns}
        col_cosmology.addCosmology('myCosmo', params)
        colcosmo=col_cosmology.setCosmology('myCosmo')
        
        if print_cosmo:
            print(colcosmo)

        Mass_bin = np.logspace(np.log10(Mmin),np.log10(Mmax), num=100)
        #Mh=Mass_bin/p.small_h
        
        Mh=Mass_bin
        mfunc = mass_function.massFunction(Mh, z, mdef = mdef, model = self.halo_model, q_out = q_out)
        
        mpc_to_m= 3.086e+22
        
        mfunc*=(mpc_to_m)**-3
                
        dndm=mfunc
        
        return Mass_bin, dndm


