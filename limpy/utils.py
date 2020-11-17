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
    """
    Comoving box size to degree converter. 
    return: boxsize in degree scale. 
    """
    
    mpc_to_m=p.mpc_to_m
    boxsize=boxsize*mpc_to_m
    dc=cosmo.D_co(z)
    theta_rad=boxsize/dc
    theta_degree=theta_rad*180.0/np.pi
    return theta_degree

def physical_boxsize_to_degree(z, boxsize):
    """
    Physical boxsize to degree converter. 
    return: boxsize in degree scale. 
    """
    
    mpc_to_m=p.mpc_to_m
    boxsize=boxsize*mpc_to_m
    da=cosmo.D_angular(z)
    theta_rad=boxsize/da
    theta_degree=theta_rad*180.0/np.pi
    return theta_degree


def degree_to_comoving_size(z, theta_degree):
    """
    This function converts simulation box from degree unit to comoving Mpc.
    """

    theta_rad=theta_degree*np.pi/180
    dc=cosmo.D_co(z)
    size=theta_rad*dc
  
    return size


def freq_to_lambda(nu):
    "frequency to wavelength converter."
    wav=p.c_in_m/nu
    return wav



def Omega_beam(theta_min, factor=2.355):
    """
    Calculates the solid angle substended by a telescope beam. 
    """
    
    theta_rad=theta_min*p.minute_to_radian
    
    return 2*np.pi*(theta_rad/factor)**2



def t_pix(theta_min, tobs_total, Ndet_eff, S_area):
    """
    Time per pixel.
    
    theta_min: the beam size in arc-min.
    tobs_total: total observing time.
    Ndet_eff: Effective number of detectors, for CCATp, Ndet_eff~20. 
    S_area: Survey area in degree^2. 
    
    return: t_pix in second. 
    
    """
    omega_beam=Omega_beam(theta_min)
    S_area_rad= S_area*(p.degree_to_radian)**2
    
    tobs_total*=3600 # hours to seconds
    res=tobs_total*Ndet_eff*omega_beam/(S_area_rad)
    return res



def V_surv(z, S_area, B_nu, line_name='CII'):
    '''
    Calculates the survey volume in MPc. 
    
    z: redshift
    lambda_line: frequncy of line emission in micrometer
    A_s: Survey area in degree**2
    B_nu: Total frequency band width resolution in GHz
    
    return: Survey volume. 
    '''
    
    nu=p.nu_rest(line_name)*p.Ghz_to_hz
    B_nu*=p.Ghz_to_hz
    Sa_rad=S_area*(p.degree_to_radian)**2

    lambda_line=freq_to_lambda(nu)
    
    y=lambda_line*(1+z)**2/cosmo.H_z(z)
    res=cosmo.D_co(z)**2*y*(Sa_rad)*B_nu
    return res*(p.m_to_mpc)**3

def nu_obs_to_z(nu_obs, line_name='CII'):
    """
    This function evaluates the redshift of a particular line emission 
    corresponding to the observed frequency. 
    
    return: redshift of line emission. 
    """
    
    
    nu_rest_line=p.nu_rest(line_name=line_name)
    assert(nu_obs<=nu_rest_line),"Observed frequency cannot be smaller than the %s rest frame frequency. In that case z will be negative, which is non physical" %(line_name)

    z=(nu_rest_line/nu_obs)-1
    return z

def V_pix(z, theta_min, delta_nu, line_name='CII'):
    '''
    z: redshift
    lambda_line: frequncy of line emission in micrometer
    theta_min: beam size in arc-min
    delta_nu: the frequency resolution in GHz
    '''
      
    theta_rad=theta_min*p.minute_to_radian
    
    nu=p.nu_rest(line_name)*p.Ghz_to_hz
    delta_nu*=p.Ghz_to_hz

    lambda_line=freq_to_lambda(nu)
    
    y=lambda_line*(1+z)**2/cosmo.H_z(z)
    res=cosmo.D_co(z)**2*y*(theta_rad)**2*delta_nu
    return res*(p.m_to_mpc)**3



def sigma_noise(theta_min, NEI, experiment='ccatp'):
    '''
    noise per pixel. 
    '''
    
    
    if(experiment=='ccatp'):
        return NEI
    
    if(experiment=='other'):
        omegab=Omega_beam(theta_min)
        return NEI/omegab
    
        
    #omega_beam=Omega_beam(theta_min)
    return NEI*4*np.pi
    
    
def P_noise(z, theta_min, delta_nu, NEI, tobs_total, Nspec_eff, S_a):
    """
    White noise of an experiment.
    z: redshift
    theta_min: beam size in arc-min
    delta_nu: frequency resolution in GHz. 
    NEI: noise equivalence impedence. 
    tobs_total: total observing time in hours. 
    Nspec_eff: effective number of detectors. 
    S_a: Survey area in degree^2. 
    """
    
    
    Pn=V_pix(z, theta_min, delta_nu)*sigma_noise(theta_min, NEI)**2/(t_pix(theta_min, tobs_total, Nspec_eff, S_a))
    return Pn


def P_noise_ccatp(nu=220):
    if(nu==220):
        res=1.2e9
        
    elif(nu==280):
        res=2e9
        
    elif(nu==350):
        res=6.3e9
        
    elif(nu==410):
        res=2.3e10
        
    return res


def N_modes(k,z, delta_k, A_s, B_nu, line_name='CII' ):
    Vs=V_surv(z, A_s, B_nu, line_name=line_name)
    res=2*np.pi*k**2*delta_k*Vs/(2*np.pi)**3
    return res
    

    