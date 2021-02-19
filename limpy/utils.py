#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:57:35 2020

@author: anirbanroy
"""
#from __future__ import division
import numpy as np
import importlib
import imp 

import cosmos as cosmos
import params as p

imp.reload(cosmos) 
imp.reload(p)


import warnings


from multiprocessing import cpu_count
THREADS = cpu_count()
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


def length_projection(z=None, dz=None, nu_obs=None, dnu=None, line_name='CII'):
    if (z is None and dz is not None) or (z is not None and dz is None):
        raise ValueError ("Specify z and dz together to calculate the projection length calculation.")
        
    if (nu_obs is None and dnu is not None) or (nu_obs is not None and dnu is None):
        raise ValueError ("Specify nu_obs and dnu together to calculate the projection length calculation.")
        
    if (z is None and dz is None) and (nu_obs is None and dz is None):
         raise ValueError ("Either specify z and dz together or nu_obs and dnu together for projection length calculation.")
        
        
    if z !=None and dz !=None:
        dco1=cosmo.D_co(z)
        dco2=cosmo.D_co(z+dz)
        res= (dco2-dco1)* p.m_to_mpc
        
    if dnu !=None and dnu !=None:
        z_obs1=nu_obs_to_z(nu_obs, line_name=line_name)
        z_obs2=nu_obs_to_z((nu_obs+dnu), line_name=line_name)
        
        dco1=cosmo.D_co(z_obs1)
        dco2=cosmo.D_co(z_obs2)
        res= (dco1-dco2)* p.m_to_mpc
 
    return res

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



def slice(datacube, ngrid, nproj, option='C'):
    """
    Produces a slice from a 3D data cube for plotting. `option' controls
    whether data cube has C or Fortran ordering.

    """
    iarr = np.zeros(ngrid*ngrid)
    jarr = np.zeros(ngrid*ngrid)
    valarr = np.zeros(ngrid*ngrid)

    counter = 0
    for i in range(ngrid):
        for j in range(ngrid):
            iarr[counter] = i
            jarr[counter] = j
            valarr[counter] = 0.0
            for k in range(nproj):
                if option=='F':
                    valarr[counter] += datacube[i+ngrid*(j+ngrid*k)]
                elif option=='C':
                    valarr[counter] += datacube[k+ngrid*(j+ngrid*i)]
            counter += 1 

    return iarr, jarr, valarr


def freq_2D(boxsize, ngrid):
    kf = 2.0*np.pi/boxsize
    kn = np.pi/(boxsize/ngrid)
    
    return kf, kn




def slice_2d(datacube, ngrid, nproj, operation='sum'):
    """
    Produces a slice from a 3D data cube for power spectra calculation.
    
    nproj: number of cells to project.
    
    operation suggestion either to "sum" or "mean" over projection cells.
    """
    
    ndim=np.ndim(datacube)
    #print("The dimension of data", ndim)
    
    if(ndim==1):
        data_cut=datacube.reshape(ngrid, ngrid, ngrid)[:, :, :nproj]
        
    
    elif(ndim==3):
        data_cut=datacube[:, :, :nproj]
        
    else:
        raise ValueError("Provide the data either in 1D or 3D data cube (in case of projection)")
        
    if operation=='sum':
    # Project number of cells along the third axis. 
        data_2d = data_cut.sum(axis=2)
    if operation=='mean':
        data_2d = data_cut.mean(axis=2)
        
    return data_2d
        
    



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
  
    
def read_grid(fname, ngrid=None):
    """
    Read a 3D grid from a dat file. 
    """
    
    
    with open(fname, 'rb') as f:
            grid = np.fromfile(f, dtype='f', count=-1)
    
    if ngrid is not None:
        return grid.reshape(ngrid, ngrid, ngrid)
    else:       
        return grid
    



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
    



from numpy.fft import fftn, ifftn, ifftshift, fftshift, fftfreq



def magnitude_grid(x, dim=None):
    if dim is not None:
        return np.sqrt(np.sum(np.meshgrid(*([x ** 2] * dim)), axis=0))
    else:
        return np.sqrt(np.sum(np.meshgrid(*([X ** 2 for X in x])), axis=0))

def myfft(x_grid, boxlength, ngrid, ndim=2, a = 0, b=2*np.pi):
    
    #volume of the box
    #V = boxlength**ndim
    
    # Volume of each cells
    cellsize=boxlength/ngrid
    V_cell=cellsize**ndim
    
    #print("V_cell", V_cell)
    
    #do fftn
    fftn_res=fftn(x_grid, axes=[0,1])
    
    ft_res=V_cell * fftshift(fftn_res , axes=[0,1])* (np.abs(b) / (2 * np.pi) ** (1 - a)) ** (ndim/2)
    
    #left_edge=np.array([-boxlength/2.0,  --boxlength/2.0])
    
    #dx = np.array([float(l) / float(n) for l, n in zip(L, N)])
    
    frequency =fftshift(fftfreq(ngrid, d=(boxlength/ngrid)))* (2 * np.pi / b)
    
    k_grid=np.add.outer(frequency[0]**2, frequency[1] ** 2)
    
    return ft_res, np.array([frequency, frequency]), k_grid
    


def myifft(x_grid, boxlength, ngrid, ndim=2, a = 0, b=2*np.pi):
    
    boxlength_k=(2 * np.pi * ngrid)/(boxlength * b)
    
    #volume of the box in k space
    V_k=boxlength_k**ndim
    
    #print("V_cell", V_k)
    
    #do ifftn
    
    ifftn_res = V_k * ifftn(x_grid, axes=[0,1]) * (np.abs(b) / (2 * np.pi) ** (1 + a)) ** (ndim/2)
    
    ift_res=ifftshift(ifftn_res, axes=[0,1])
    
    #left_edge=np.array([-boxlength/2.0,  --boxlength/2.0])
    
    #dx = np.array([float(l) / float(n) for l, n in zip(L, N)])
    
    print( ift_res)
    
    frequency =fftshift(fftfreq(ngrid, d=(boxlength_k/ngrid)))* (2 * np.pi / b)
    
    k_grid=np.add.outer(frequency[0]**2, frequency[1] ** 2)
    
    return ift_res, np.array([frequency, frequency]), k_grid


def powerspectra_2d(x_grid, boxlength, ngrid, project_length=None, nproj=None, a=1, b=1, ndim=2, 
                    volume_normalization=False, ignore_zero_mode=True, bins_num=None, y_grid=None,
                   remove_shotnoise=False):
    
    Vbox=boxlength**ndim
    cellsize=boxlength/ngrid
    
    
    if (project_length is not None) and (nproj is None):
        
        if(project_length != None):
            nproj=int(round(project_length/cellsize))
            #print("The number of cells to be projected=", nproj)
        if(nproj != None):
            nproj=nproj
            #print("The number of cells to be projected=", nproj)
            
        if x_grid is not None:
            g_xi = slice_2d(x_grid, ngrid, nproj)
        
        if y_grid is not None:
            g_yi= slice_2d(y_grid, ngrid, nproj)
            
    if (project_length is None) and (nproj is None):
        if(len(np.shape(x_grid)) != ndim):
            raise ValueError ("X_grid should be a 2D grid.")
        
        if x_grid is not None:
            g_xi = x_grid
        
        if y_grid is not None:
            g_yi= y_grid
        
        #raise ValueError("Specify a projection length along the radial direction for 2D power spectrum calculation")
        
    ft_x, fq, kgrid_x = myfft(g_xi, boxlength, ngrid, a=a, b=b)
    
    if y_grid is not None:
        ft_y = myfft(g_yi, boxlength, ngrid, a=a, b=b)[0]
    else: 
        ft_y=ft_x
    
    
    P = np.real(ft_x * np.conj(ft_y) / Vbox**2)
            
    if volume_normalization:
        P*=Vbox
    
    if len(fq) == len(P.shape):
        fq = magnitude_grid(fq)
        
    if bins_num is not None:
        bins_num=bins_num 
    else:
        N=[ngrid]*ndim
        bins_num=bins = int(np.product(N[:ndim]) ** (1. / ndim) / 2.2)
        
        
        
    bins = np.linspace(fq.min(), fq.max(), (bins_num+1))    
    bin_index = np.digitize(fq.flatten(), bins)
    binweight=np.bincount(bin_index, minlength=len(bins)+1)[1:-1]
    
    # Do average over fields.    
    #weights=np.real(P.flatten())
    
    #print("the weight shape", np.shape(weights))
    
    real_part = np.bincount(bin_index, weights=np.real(P.flatten()), minlength=len(binweight)+2)[1:-1] / binweight
    if(P.dtype.kind=='c'):
        imaginary_part = 1j * np.bincount(bin_index, weights=np.imag(P.flatten()), minlength=len(binweight)+2)[1:-1] / binweight
    else:
        imaginary_part=0
    
    field_average= real_part + imaginary_part
    
    #print("the weight shape", np.shape(weights))
    
    res = list(field_average)
    
    if remove_shotnoise:
        N1=x_grid.shape[0]
        if y_grid is not None:
            N2= y_grid.shape[0]
        else: 
            N2=N1
        res[0] -= np.sqrt(Vbox ** 2 / N1 / N2) 
    
    return bins[1:], res
    


def make_hlist_ascii_to_npz(hlist_path_ascii, filename=None):
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
    data=np.loadtxt(hlist_path_ascii)  
    
    # save the file in npz format either in mentioned filename or in original ascii filename
    if filename:
        np.savez(filename,m=data[:,0],x=data[:,1],y=data[:,2],z=data[:,3])
    else:
        np.savez(hlist_path_ascii,m=data[:,0], x=data[:,1], y=data[:,2], z=data[:,3])
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



def make_grid_from_halocat(halo_catalouge, boxlength, ngrid, ndim, filetype='dat'):
    mh, halo_cm=make_halocat(halo_catalouge,filetype=filetype, boxsize=boxlength)
    
    halo_cm=halo_cm.reshape(3, len(mh))
    
    cellsize=(boxlength/ngrid)
    
    bins = cellsize*np.arange(0,ngrid,1)
    
    bin_index_x=np.digitize(halo_cm[0], bins)
    bin_index_y=np.digitize(halo_cm[1], bins)
    bin_index_z=np.digitize(halo_cm[2], bins)
    
    binweight_x=np.bincount(bin_index_x, minlength=len(bins)+1)[1:-1]
    binweight_y=np.bincount(bin_index_y, minlength=len(bins)+1)[1:-1]
    binweight_z=np.bincount(bin_index_z, minlength=len(bins)+1)[1:-1]
    
    return binweight_x,  binweight_y,  binweight_z
    


def cic(x):
    dx, f = np.modf(x)
    return int(f), (1.0-dx, dx)
  
def grid(posarr, weight, boxlength, ngrid, ndim=3):
    cell_size=boxlength/ngrid
    Vcell=cell_size**ndim
    
    n = ngrid
    gridarr = np.zeros(n**3)
    for i in range(weight.size):
        x = posarr[3*i]/cell_size
        y = posarr[3*i+1]/cell_size
        z = posarr[3*i+2]/cell_size
        fx, cx = cic(x)
        fy, cy = cic(y)
        fz, cz = cic(z)
        for i1 in range(2):
            j1 = i1+fx if i1+fx < n else 0  
            #print(j1)
            for i2 in range(2):
                j2 = i2+fy if i2+fy < n else 0 
                for i3 in range(2):
                    j3 = i3+fz if i3+fz < n else 0 
                    gridarr[j3+n*(j2+n*j1)] += (cx[i1]*cy[i2]*cz[i3]*
                                                weight[i]/Vcell)
    return gridarr 




