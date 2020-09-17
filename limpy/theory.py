#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:55:42 2020

@author: anirbanroy
"""

import numpy as np
from hmf import MassFunction, transfer
import numpy as np
import params as p
import matplotlib.pyplot as plt
import lline as ll
import cosmos
from scipy.integrate import simps
from scipy.interpolate import interp2d, interp1d



"""
# IF we want to keep same cosmological parameters for the code and HMFcal
"""
#from hmf import cosmo as cosmo_hmf
#my_cosmo = cosmo_hmf.Cosmology()
#my_cosmo.update(cosmo_params={"H0":71,"Om0":0.281,"Ode0":0.719,"Ob0":0.046})

def hmf(z, Mmin=Mmin, Mmax=Mmax, output_quantity='dndm'):
    '''Shet, Mo &  Tormen 2001'''
    mf=MassFunction(z=z,Mmin=np.log10(Mmin), Mmax=np.log10(Mmax), hmf_model= Halo_model)
    #return mf.m, mf.dndlog10m
    if(output_quantity=='dndm'):
        hm, dndm= mf.m/small_h, mf.dndm * small_h**4*(mpc_to_m)**-3

        return hm, dndm
    if(output_quantity=='nu'):

        return mf.m/small_h, mf.nu



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
    onepz, SFR =np.loadtxt(SFR_filename,unpack=True, usecols=(0,2))

    # scale factor to redshift
    z=onepz-1

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
    scale_fac, SFR, error_SFR_up, error_SFR_down=np.loadtxt(SFR_filename,unpack=True)
    # scale factor to redshift
    z=1.0/scale_fac-1.0

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
    data=np.loadtxt(hlist_path_ascii,usecols=(0,10,17,18,19))

    # save the file in npz format either in mentioned filename or in original ascii filename
    if saved_filename:
        np.savez("hlist_0.12650.npz",a=data[:,0],m=data[:,1],x=data[:,2],y=data[:,3],z=data[:,4])
    else:
        np.savez(hlist_path_ascii,a=data[:,0],m=data[:,1],x=data[:,2],y=data[:,3],z=data[:,4])
    return


def make_halocat(halo_file, filetype='npz',boxsize=160):
    """
    reads the mass and co-ordinates of halos from a npz file.

    Input: halo file in npz format

    Returns: m and (x,y,z)
    """
    if filetype=='npz':
        fn=np.load(halo_file)
        #halomass and x,y,z are read in the following format
        halomass, halo_x, halo_y, halo_z = fn['m'], fn['x'],fn['y'],fn['z']

    if filetype=='dat':
        halomass, halo_x, halo_y, halo_z=np.loadtxt(halo_file,unpack=True)
        halomass, halo_x, halo_y, halo_z = halomass, halo_x*boxsize, halo_y*boxsize, halo_z *boxsize


    # stack the co-ordinates together
    halo_cm=np.column_stack([halo_x,halo_y,halo_z])
    del halo_x
    del halo_y
    del halo_z
    halo_cm=halo_cm.flatten()

    return halomass, halo_cm


def sfr_to_lcp_scatter(z, sfr,a_off=p.default_lcp_scatter_params['a_off'],a_std=p.default_lcp_scatter_params['a_std'],
                       b_off=p.default_lcp_scatter_params['b_off'],b_std=p.default_lcp_scatter_params['b_std']):

    """
    Calculates lumiosity of the CII lines from SFR assuming a 3\sigma Gussian scatter. The parameter values for the scattered relation
    is mentioned in defalut_params modeule.

    Input: z and sfr

    return: luminosity of CII lines in log scale
    """
    if np.isscalar(sfr)==True:
        sfr=np.atleast_1d(sfr)

    sfr_len=len(sfr)
    log_L_cp=np.zeros(sfr_len)
    for i in range(sfr_len):
        a= np.random.normal(a_off,a_std)
        b= np.random.normal(b_off,b_std)
        log_L_cp[i]=(a+b*np.log10(sfr[i]))

    return log_L_cp



def sfr_to_lcp_nonscatter(z, sfr,a_off=p.default_lcp_scatter_params['a_off'],b_off=p.default_lcp_scatter_params['b_off']):
    """
    This function returns luminosity of CII lines in the unit of L_sun. This does not include the scatter, rather
    this is the mean relation.

    Values of fitting parameters are taken from Eq (1) from Chung et al. 2020 (arxiv: 1812.08135)

    We write two equations as
    \alpha_z=a-b*z
    \beta_z=c-d*z
    values are mentioned
    """
    a,b=a_off, b_off

    if np.isscalar(sfr)==True:
        sfr=np.atleast_1d(sfr)

    sfr_len=len(sfr)
    log_L_cp=np.zeros(sfr_len)

    for i in range(sfr_len):
        log_L_cp[i]=(a+b*np.log10(sfr[i]))

    return log_L_cp


def sfr_to_lcp_nonscatter_chung(z, sfr):
    """
    This function returns luminosity of CII lines in the unit of L_sun. This does not include the scatter, rather
    this is the mean relation.

    Values of fitting parameters are taken from Eq (1) from Chung et al. 2020 (arxiv: 1812.08135)

    We write two equations as
    \alpha_z=a-b*z
    \beta_z=c-d*z
    values are mentioned
    """
    a,b,c,d=p.default_lcp_chung_params['a'],p.default_lcp_chung_params['b'],\
    p.default_lcp_chung_params['c'],p.default_lcp_chung_params['d']
    alpha_z= a-b*z
    beta_z= c- d*z
    log_lcp=alpha_z*np.log10(sfr)+beta_z

    return log_lcp

def mhalo_to_lco_prime_fit(Mhalo,z, nu_rest):
    Mhalo=np.array(Mhalo)
    #assert z<=3, "LCO-Mhalo relation is valid for redshift between 0 and 3."
    M10=4.17e12
    M11=-1.17
    N10=0.0033
    N11=0.04
    b10=0.95
    b11=0.48
    y10=0.66
    y11=-0.33
    M1_z=10**(np.log10(M10)+(M11*z)/(z+1))
    N_z=N10+(N11*z)/(z+1)
    b_z=b10+(b11*z)/(z+1)
    y_z=y10+(y11*z)/(z+1)

    Lco_prime=(2*N_z*Mhalo)/((Mhalo/M1_z)**-b_z+(Mhalo/M1_z)**y_z)

    return Lco_prime



def mhalo_to_lcp_fit(Mhalo,z):
    Mhalo=np.array(Mhalo)
    M1=2.39e-5
    N1=4.19e11
    alpha=1.79
    beta=0.49

    F_z=((1+z)**2.7/(1+((1+z)/2.9)**5.6))**alpha
    Lcii=F_z*((Mhalo/M1)**beta)*np.exp(-N1/Mhalo)

    return Lcii

def lco_prime(Mhalo, z, line_name='CO10'):
    #FIXME please check the relation for J-lader
    line_name_len=len(line_name)
    if(line_name_len==4):
         J_lader=int(line_name[2])
    elif(line_name_len==5 or line_name_len==6):
        J_lader=int(line_name[2:4])
    Lco_prime=mhalo_to_lco_prime_fit(Mhalo, z, nu_rest_CO10)
    L_line=(J_lader)**3*Lco_prime

    return L_line


def lco_prime_to_lco(Mhalo, z, line_name='CO10'):
    line_name_len=len(line_name)
    if(line_name_len==4):
         J_lader=int(line_name[2])
    elif(line_name_len==5 or line_name_len==6):
        J_lader=int(line_name[2:4])

    Lco_prime=mhalo_to_lco_prime_fit(Mhalo, z, nu_rest_CO10)
    L_line=4.9e-5*(J_lader)**3*Lco_prime  # Equation 4 of  arxiv:1706.03005

    return L_line

def mhalo_to_lline(Mhalo, z, line_name='CII'):
    if(line_name=="CII"):
        L_line=mhalo_to_lcp_fit(Mhalo, z)
    if(line_name[0:2]=="CO"):
        L_line=lco_prime_to_lco(Mhalo, z, line_name=line_name)

    return L_line


def T_line(z,nu_rest_line,line_name="CII",fduty=1.0):

    mass_bin, dndm= hmf(z)

    L_line= mhalo_to_lline(mass_bin, z, line_name=line_name)
    L_line*=Lsun

    nu_rest_line_Hz=nu_rest_line*Ghz_to_hz
    integrand=dndm * L_line
    integration=simps(integrand, mass_bin)
    factor=fduty*(c_in_m**3/8.0/np.pi)*((1+z)**2/(kb_si*nu_rest_line_Hz**3*cosmo.H_z(z)))
    result=factor*integration
    return result


def I_nu(z,nu_rest_line,line_name="CII", fduty=1.0):
    T=T_line(z,nu_rest_line,line_name=line_name,fduty=fduty)
    nu_rest_line_Hz=nu_rest_line*Ghz_to_hz
    I=(2*kb_si*T/c_in_m**2)*(nu_rest_line_Hz**2/(1+z)**2)

    return I/(jy_unit)


def P_shot(z,line_name='CII'):
    #mass_range=np.logspace(np.log10(Mmin), np.log10(Mmax),num=500)
    mass_bin, dndm= hmf(z)
    L_line= mhalo_to_lline(mass_bin, z, line_name=line_name)
    L_line*=Lsun

    integrand_numerator=dndm*(L_line)**2/(mpc_to_m)**-3 #in Mpc unit
    integrand_denominator=dndm*(L_line)/(mpc_to_m)**-3 #in Mpc unit
    int_numerator=simps(integrand_numerator, mass_bin)
    int_denominator=simps(integrand_denominator, mass_bin)

    return int_numerator/int_denominator**2


def nu(m,z):
    m_n, nu=hmf(z, output_quantity='nu')
    nu_int=interp1d(m_n, nu)

    return  nu_int(m)


def bias_nu(nu, delta_v=200.):
        y = np.log10(delta_v)
        A = 1.0 + 0.24*y*np.exp(-(4./y)**4.)
        a = 0.44*y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4.)
        c = 2.4

        return (1. - A*nu**a/(nu**a + delta_c**a) + B*nu**b + C*nu**c)


def bias_dm(m,z):
    nu_m=nu(m,z)

    return bias_nu(nu_m, delta_v=200.)

def b_line(z, line_name='CII'):
    mass_bin, dndm= hmf(z)
    L_line= mhalo_to_lline(mass_bin, z, line_name=line_name)
    L_line*=Lsun
    integrand_numerator=dndm*(L_line)*bias_dm(mass_bin, z)
    integrand_denominator=dndm*(L_line)
    int_numerator=simps(integrand_numerator, mass_bin)
    int_denominator=simps(integrand_denominator, mass_bin)

    return int_numerator/int_denominator


def Pk_line(k,z,nu_rest_line,fduty=1.0,line_name='CII',label='total', pk_unit='temperature'):
    if(pk_unit=='intensity'):
        I_nu_square=I_nu(z,nu_rest_line, line_name=line_name)**2
        pk_lin=cosmo.pk_camb(k,z)
        if(label=='total'):
            res=I_nu_square*(b_line(z, line_name=line_name)**2*pk_lin+P_shot(z, line_name=line_name))
        if(label=='clustering'):
            res=I_nu_square*(b_line(z, line_name=line_name)**2*pk_lin)
        if(label=='shot'):
            res=I_nu_square*(P_shot(z, line_name=line_name))

        d2k=k**3*res/2/np.pi**2# in (Jy/sr)^2 unit
        return d2k


    if(pk_unit=='temperature' or pk_unit=='muk' ):
        T_line_square=T_line(z,nu_rest_line, line_name=line_name, fduty=fduty)**2
        pk_lin=cosmo.pk_camb(k,z)
        if(label=='total'):
            res=T_line_square*(b_line(z, line_name=line_name)**2*pk_lin+P_shot(z, line_name=line_name))
        if(label=='clustering'):
            res=T_line_square*(b_line(z, line_name=line_name)**2*pk_lin)
        if(label=='shot'):
            res=T_line_square*(P_shot(z, line_name=line_name))


        d2k=k**3*res/2/np.pi**2 #in K unit
        return d2k*1e12  #in muK
