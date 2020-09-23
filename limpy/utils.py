#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:57:35 2020

@author: anirbanroy
"""
from __future__ import division
import numpy as np
import cosmos 
import params as p

cosmo=cosmos.cosmo()

def volume_box(boxsize):
    """
    Total volume of the simulation box. 
    unit: boxsize in Mpc
    return: volume in Mpc^3
    """
    return boxsize**3 #in mpc

def volume_cell(boxsize,ngrid):
    """
    The volume of a cell in a simulation box.
    unit: boxsize in Mpc, ngrid: 1nt
    return: volume in Mpc^3
    """
    clen= boxsize/ngrid   #length of a cell
    return clen**3   # in Mp

def comoving_boxsize_to_degree(z, boxsize):
    #boxsize in Mpc
    mpc_to_m=p.mpc_to_m
    boxsize=boxsize*mpc_to_m
    dc=cosmo.D_co(z)
    theta_rad=boxsize/dc
    theta_degree=theta_rad*180.0/np.pi
    return theta_degree

def physical_boxsize_to_degree(z, boxsize):
    #boxsize in Mpc
    mpc_to_m=p.mpc_to_m
    boxsize=boxsize*mpc_to_m
    da=cosmo.D_angular(z)
    theta_rad=boxsize/da
    theta_degree=theta_rad*180.0/np.pi
    return theta_degree


def degree_to_comoving_size(z, theta_degree):
    #boxsize in Mpc
    theta_rad=theta_degree*np.pi/180
    dc=cosmo.D_co(z)
    size=theta_rad*dc
  
    return size


def freq_to_lambda(nu):
    wav=p.c_in_m/nu
    return wav



def Omega_beam(theta_arcmin, factor=2.355):
    'return in arc-min^2 unit'
    
    return 2*np.pi*(theta_arcmin/factor)**2



def Omegab(theta_arcmin, factor=2.355):
    'return in arc-min^2 unit'
    return 2*np.pi*(theta_arcmin/factor)**2




def t_pix(theta_arcmin, tobs_total, Ndet_eff, S_area):
    omega_beam=Omega_beam(theta_arcmin, factor=2.355)
    res=tobs_total*Ndet_eff*omega_beam/(S_area*3600)
    return res



def V_surv(z, A_s, B_nu, line_name='CII'):
    '''
    z: redshift
    lambda_line: frequncy of line emission in micrometer
    A_s: Survey area in degree**2
    B_nu: Total frequency band width resolution in GHz
    '''
    
    nu=p.nu_rest(line_name)*p.Ghz_to_hz
    B_nu*=p.Ghz_to_hz
    As_rad=A_s*(np.pi/180)**2

    lambda_line=freq_to_lambda(nu)
    
    y=lambda_line*(1+z)**2/cosmo.H_z(z)
    res=cosmo.D_co(z)**2*y*(As_rad)*B_nu
    return res*(p.m_to_mpc)**3




def V_pix(z, theta_min, delta_nu, line_name='CII'):
    '''
    z: redshift
    lambda_line: frequncy of line emission in micrometer
    theta_min: beam size in arc-min
    delta_nu: the frequency resolution in GHz
    '''
      
    theta_d=theta_min/60
    theta_rad=theta_d*np.pi/180.0
    
    nu=p.nu_rest(line_name)*p.Ghz_to_hz
    delta_nu*=p.Ghz_to_hz

    lambda_line=freq_to_lambda(nu)
    
    y=lambda_line*(1+z)**2/cosmo.H_z(z)
    res=cosmo.D_co(z)**2*y*(theta_rad)**2*delta_nu
    return res*(p.m_to_mpc)**3


def NEI_to_NEFD(NEI, Ndet):
    return NEI/np.sqrt(Ndet)

def sigma_noise(theta_min, NEI, Ndet):
    NEFD=NEI_to_NEFD(NEI, Ndet)
    omega_beam=Omega_beam(theta_min)
    return NEFD/omega_beam
    
    
def P_noise(z, theta_min, delta_nu, NEI, tobs_total, Nspec_eff, S_a,  Ndet):
    Pn=V_pix(z, theta_min, delta_nu)*sigma_noise(theta_min, NEI, Ndet)**2/(t_pix(theta_min, tobs_total, Nspec_eff, S_a))
    return Pn


def P_noise_ccatp():
    return 2e9


def N_modes(k,z, delta_k, A_s, B_nu, line_name='CII' ):
    Vs=V_surv(z, A_s, B_nu, line_name=line_name)
    res=2*np.pi*k**2*delta_k*Vs/(2*np.pi)**3
    return res
    

    