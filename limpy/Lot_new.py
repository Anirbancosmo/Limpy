#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division #changes the division operator?
import numpy as np
import params as p
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import utils  
from matplotlib import cm
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, Gaussian1DKernel
from astropy.modeling.models import Gaussian2D

"""
Created on Fri Jul 17 13:29:46 2020

@author: reustudent-Dariannette Valentin

"""
"""OIII lines"""

def read_SFR(SFR_filename):
    #Reading columns (SFR parameters) from file
    scale_fac, SFR, error_SFR_up, error_SFR_down=np.loadtxt(SFR_filename,unpack=True) #I dont understand the last column?
    
    # scale factor to redshift
    z=1.0/scale_fac-1.0
    
    return z, SFR, error_SFR_up, error_SFR_down


def make_hlist_ascii_to_npz(hlist_path_ascii, saved_filename=None):
    
    # reads only scale factor, halomass,x,y and z from the ascii file used in Universe machine
    data=np.loadtxt(hlist_path_ascii,usecols=(0,10,17,18,19))  
    
    # save the file in npz format either in mentioned filename or in original ascii filename
    if saved_filename:
        np.savez("hlist_0.12650.npz",a=data[:,0],m=data[:,1],x=data[:,2],y=data[:,3],z=data[:,4])
    else:
        np.savez(hlist_path_ascii,a=data[:,0],m=data[:,1],x=data[:,2],y=data[:,3],z=data[:,4])
    return


def make_halocat(halo_file, filetype='npz',boxsize=160):
    ''' Returns m and (x,y,z)'''
    
    if filetype == 'npz':
        fn = np.load(halo_file)
        #halomass and x,y,z are read in the following format
        halomass, halo_x, halo_y, halo_z = fn['m'], fn['x'], fn['y'], fn['z'] ###are parameters in halo file named exactly like this?
            
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


def sfr_to_lot_scatter(z, sfr,a_off=p.default_lot_scatter_params['a_off'],a_std=p.default_lot_scatter_params['a_std'],
                       b_off=p.default_lot_scatter_params['b_off'],b_std=p.default_lot_scatter_params['b_std']):
    
    """
    Calculates lumiosity of the OIII lines from SFR assuming a 3\sigma Gussian scatter. The parameter values for the scattered relation
    is mentioned in defalut_params module. 
    
    Input: z and sfr
    
    return: luminosity of OIII lines in log scale 
    """
    if np.isscalar(sfr)==True:
        sfr=np.atleast_1d(sfr) ####Convert inputs to arrays with at least one dimension. Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.
    
    sfr_len=len(sfr)
    log_L_ot=np.zeros(sfr_len)
    for i in range(sfr_len):
        a= np.random.normal(a_off,a_std)
        b= np.random.normal(b_off,b_std)
        log_L_ot[i]=(a+b*np.log10(sfr[i]))
    return log_L_ot

   

def sfr_to_lot_nonscatter(z, sfr,a_off=p.default_lot_scatter_params['a_off'],b_off=p.default_lot_scatter_params['b_off']):
    """
    This function returns luminosity of OIII lines in the unit of L_sun. This does not include the scatter, rather
    this is the mean relation. 
    
    Values of fitting parameters are taken from .......
    
    We write two equations as
    \alpha_z=a-b*z
    \beta_z=c-d*z
    values are mentioned 
    """
    a,b=a_off, b_off
  
    if np.isscalar(sfr)==True:
        sfr=np.atleast_1d(sfr)
    
    sfr_len=len(sfr)
    log_L_ot=np.zeros(sfr_len)
    
    for i in range(sfr_len):
        log_L_ot[i]=(b*np.log10(sfr[i])+a)
    return log_L_ot


'''def sfr_to_lcp_nonscatter_chung(z, sfr):
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
    return log_lcp '''


def mhalo_to_sfr(logMh):
    """
    Returns the SFR history for discrete values of halo mass.
    
    logMh values should be integrer values between 11 to 15 
    (check the resonable redshift so that data is ava)
    
    
    #TODO: Make it a continuous function so that we can interpolate smoothly 
    between Mmin=10^9 to 10^15
    """
    
    sfr_filepath='../data/sfh_z0_z8/sfr/'
        
    if logMh==11:
        sfr_fname=sfr_filepath+'sfr_corrected_11.0.dat'
    if logMh==12:
        sfr_fname=sfr_filepath+'sfr_corrected_12.0.dat'
        
    if logMh==13:
        sfr_fname=sfr_filepath+'sfr_corrected_13.0.dat'
        
    if logMh==14:
        sfr_fname=sfr_filepath+'sfr_corrected_14.0.dat'
        
    if logMh==15:
        sfr_fname=sfr_filepath+'sfr_corrected_15.0.dat'           
        
    z_sfr, SFR, error_SFR_up, error_SFR_down= read_SFR(sfr_fname)
    
    return z_sfr, SFR, (SFR-error_SFR_down), (SFR+error_SFR_up)


def mhalo_to_lot(z,logMh, kind='mean',use_scatter=True):
    """
    this function returns luminosity of OIII lines in the unit of L_sun.
    Kind optitions takes the SFR: mean , up (1-sigma upper bound) and
    down (1-sigma lower bound)
    """
    mhlen=len(logMh)
    result=np.zeros(mhlen)
    
    log_lot_low=p.default_lot_dummy_values['log_lot_low']
    for i in range(mhlen):
        logMh_val=logMh[i]
        
        if logMh_val<11:
            lot = log_lot_low
            
        elif(logMh_val>=11):
            z_sfr, SFR_mean, SFR_down, SFR_up=mhalo_to_sfr(logMh_val)
            SFR_mean = np.interp(z,z_sfr,SFR_mean)
            SFR_up = np.interp(z,z_sfr,SFR_up)
            SFR_down = np.interp(z,z_sfr,SFR_down)
            
            if(use_scatter==True and kind=='mean'):
                lot = sfr_to_lot_scatter(z,SFR_mean)
            if(use_scatter==True and kind=='up'):
                lot = sfr_to_lot_scatter(z,SFR_up)
            if(use_scatter==True and kind=='down'):
                lot = sfr_to_lot_scatter(z,SFR_down)
                
            if(use_scatter==False and kind=='mean'):
                lot = sfr_to_lot_nonscatter(z,SFR_mean)
            if(use_scatter==False and kind=='up'):
                lot = sfr_to_lot_nonscatter(z,SFR_up)
            if(use_scatter==False and kind=='down'):
                lot = sfr_to_lot_nonscatter(z,SFR_down)
            
        result[i]=lot
        
    return result

####need to understand from here and down

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


def plot_sfr_mhalo(Mhalo_array_logscale,colorlist=None,figname=None):
    
    colorlist=['Crimson','darkgreen','C1','blue','C5','C9']
    lw_def=3.0
    font_legend_def=18
    alpha_def=0.3
    
    x_min_def=0.0
    x_max_def=8.0
    y_min_def=0.0
    y_max_def=1000
    
    sfr_filepath='../data/sfh_z0_z8/sfr/'
    
    
    if np.isscalar(Mhalo_array_logscale)==True:
        Mhalo_array_logscale=np.atleast_1d(Mhalo_array_logscale)
        
    for i in range(len(Mhalo_array_logscale)):
        fname=sfr_filepath+'sfr_corrected_%2.1f.dat' %(Mhalo_array_logscale[i])
        z,sfr,err_up, err_down=read_SFR(fname)
        
        #plot
        if colorlist:
            plt.plot(z,sfr,lw=lw_def,color=colorlist[i],label=r"$M_{halo}=10^{%d}\,M_\odot$" %Mhalo_array_logscale[i])
            plt.fill_between(z, sfr-err_down, sfr+err_up,color=colorlist[i],label='',alpha=alpha_def)
        else:
            plt.plot(z,sfr,lw=lw_def,color=colorlist[i],label=r"$M_{halo}=10^{%d}\,M_\odot$" %Mhalo_array_logscale[i])
            plt.fill_between(z, sfr-err_down, sfr+err_up,color=colorlist[i],label='',alpha=alpha_def)
      

    plt.yscale("log")
    plt.xlim(x_min_def,x_max_def)
    plt.ylim(y_min_def,y_max_def)
    
    plt.ylabel(r"$\mathrm{SFR}\,\,(M_\odot/\mathrm{yr})$")
    plt.xlabel(r'$z$')
    
    plt.legend(loc=0, frameon=False, fontsize=font_legend_def)
        
    if figname:
        plt.savefig("%s" %(figname))
    else:
        plt.savefig("z_sfr_mhalo.png")
        

def save_luminosity_slice(boxsize, ngrid, nproj,halocat_file,halo_redshift,halo_cutoff_mass_log=11, use_scatter=True,save_unit='degree',saved_file_name=None):
    #low_mass_log=0.0
    cellsize = boxsize/ngrid
    
    halomass, halo_cm=make_halocat(halocat_file,filetype='dat',boxsize=boxsize)
    
    nhalo=len(halomass)
    # Overplot halos 
    x_halos = halo_cm[range(0,nhalo*3,3)]
    y_halos = halo_cm[range(1,nhalo*3,3)]
    z_halos = halo_cm[range(2,nhalo*3,3)]
   
    print('Minimum halo mass:', halomass.min())
    print('Maximum halo mass:', halomass.max())
   
    #halomass_filter=halomass
    logmh=np.log10(halomass)
    logmh=np.array([int(logmh[key]) for key in range(nhalo)])
    
  
    z_max = nproj*cellsize # See slice() above

    mask = z_halos < z_max
    x_halos = x_halos[mask]
    y_halos = y_halos[mask]
    halomass_slice = logmh[mask]
    

   
    mass_cut=halomass_slice >= halo_cutoff_mass_log
    halomass_slice_cut=halomass_slice[mass_cut]
    x_halos_cut= x_halos[mass_cut]
    y_halos_cut= y_halos[mass_cut]
    
    lot=mhalo_to_lot(halo_redshift, halomass_slice_cut, kind='mean',use_scatter=use_scatter)
    
    xdegree=utils.boxsize_to_degree(halo_redshift,x_halos_cut )
    ydegree=utils.boxsize_to_degree(halo_redshift,y_halos_cut)
    
    if(saved_file_name==None):
        fname=("luminosity_OIII_nproj_%d_z%1.2f" %(nproj, halo_redshift))
    else:
        fname=saved_file_name
    
    if(save_unit=='mpc'):
        np.savez(fname,x=x_halos_cut,y=y_halos_cut, luminosity=lot)
    if(save_unit=='degree'):
        np.savez(fname,x=xdegree,y=ydegree, luminosity=lot)


def calc_luminosity(boxsize, ngrid, nproj,halocat_file,halo_redshift,halo_cutoff_mass_log=11, use_scatter=True, unit='degree'):
    '''
    Calculate luminosity for input parameters
    '''
    #low_mass_log=0.0
    cellsize = boxsize/ngrid
    
    halomass, halo_cm=make_halocat(halocat_file,filetype='dat',boxsize=boxsize)
    
    nhalo=len(halomass)
    # Overplot halos 
    x_halos = halo_cm[range(0,nhalo*3,3)]
    y_halos = halo_cm[range(1,nhalo*3,3)]
    z_halos = halo_cm[range(2,nhalo*3,3)]
   
    print('Minimum halo mass:', halomass.min())
    print('Maximum halo mass:', halomass.max())
   
    #halomass_filter=halomass
    logmh=np.log10(halomass)
    logmh=np.array([int(logmh[key]) for key in range(nhalo)])
    
  
    z_max = nproj*cellsize # See slice() above

    mask = z_halos < z_max
    x_halos = x_halos[mask]
    y_halos = y_halos[mask]
    halomass_slice = logmh[mask]
    

   
    mass_cut=halomass_slice >= halo_cutoff_mass_log
    halomass_slice_cut=halomass_slice[mass_cut]
    x_halos_cut= x_halos[mass_cut]
    y_halos_cut= y_halos[mass_cut]
    
    lot=mhalo_to_lot(halo_redshift, halomass_slice_cut, kind='mean',use_scatter=use_scatter)
    
    xdegree=utils.boxsize_to_degree(halo_redshift,x_halos_cut )
    ydegree=utils.boxsize_to_degree(halo_redshift,y_halos_cut)
    
    if(unit=='mpc'):
        return x_halos_cut, y_halos_cut, lot
    if(unit=='degree'):
        return xdegree, ydegree, lot


def plot_slice(boxsize, ngrid, nproj, dens_gas_file, halocat_file,halo_redshift,halo_cutoff_mass_log=11, use_scatter=True,
               density_plot=False, halo_overplot=False, plot_lines=False, tick_label='mpc'):
    """
    Plot a slice of gas density field and overplot the distribution of
    haloes in that slice.
    
    boxsize: size of box in Mpc
    ngrid: number of grids along the one axis of simulation box
    nproj: cells to project (values could range from 1 to the ngrid)
    halocat_file: path to halo catalogue file (full path) 
    halo_redshift: redshift of halos
    halo_cutoff_mass_log: cutt off mass of the halos
    density_plot: If true, plot the density distribution
    halo_plot: If True, plot halos
    
    tick_label=either 'mpc' or degree
    
    """
    global dens_gas
    
    low_mass_log=0.0
    #z_halos=6.9  #Calculated from the scale factor of halo catalouge
    
    cellsize = boxsize/ngrid
        
    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
   
   
    if density_plot:
        # Plot gas density 
        with open(dens_gas_file, 'rb') as f:
            dens_gas = np.fromfile(f, dtype='f', count=-1)
            #dens_gas=dens_gas+1.0
            #dens_gas=dens_gas.reshape(ngrid**3,)
            #rhobar = np.mean(dens_gas)
       
        
        i, j, val = slice(dens_gas, ngrid,nproj)
       #val = val/(rhobar*nproj)
    
        cellsize = boxsize/ngrid
        i *= cellsize
        j *= cellsize 
    
    
        s = plt.scatter(i, j, c=val, s=10, marker='s',
                       edgecolor='none', rasterized=True,
                       cmap=plt.cm.gist_yarg)
        
                
        if(tick_label=='mpc'):
            ax.set_xlim(0,boxsize)
            ax.set_ylim(0,boxsize)
            plt.xlabel('cMpc')
            plt.ylabel('cMpc')
            
        elif(tick_label=='degree'):    
            ax.set_xlim(0,boxsize)
            ax.set_ylim(0,boxsize)
            
            xmin=0
            ymin=0
            xmax=ymax=utils.boxsize_to_degree(halo_redshift, boxsize)
            
            N=4
            xtick_mpc=ytick_mpc=np.linspace(0, boxsize, N)
            
            custom_yticks = np.round(np.linspace(ymin, ymax, N,dtype=float),1)
            
            ax.set_yticks(ytick_mpc)
            ax.set_yticklabels(custom_yticks)
            
            custom_xticks = np.round(np.linspace(xmin, xmax, N,dtype=float),1)
            ax.set_xticks(xtick_mpc)
            ax.set_xticklabels(custom_xticks)
            
            plt.xlabel(r'$X\,(\mathrm{degree})$')
            plt.ylabel('$Y\,(\mathrm{degree})$')
   

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(s, cax=cax)
        cb.set_label(r'$\Delta_{\rho}$',
                     labelpad=1)
       
        cb.solids.set_edgecolor("face")
        ax.set_aspect('equal', 'box')
        
        
    
    if halo_overplot:
        
        # Load density file
        with open(dens_gas_file, 'rb') as f:
            dens_gas = np.fromfile(f, dtype='f', count=-1)
            #dens_gas=dens_gas+1.0
            #dens_gas=dens_gas.reshape(ngrid,ngrid,ngrid)
            #rhobar = np.mean(dens_gas)
        
        # slice the data cube
        i, j, val = slice(dens_gas, ngrid,nproj)
    
    
        cellsize = boxsize/ngrid
        i *= cellsize
        j *= cellsize 
               

  
        s = plt.scatter(i, j, c=val, s=10, marker='s',
                       edgecolor='none', rasterized=True,
                       cmap=plt.cm.viridis, vmax=15, vmin=-15 )
        
   
    
        halomass, halo_cm=make_halocat(halocat_file,filetype='dat',boxsize=boxsize)
        
        nhalo=len(halomass)
        # Overplot halos 
        x_halos = halo_cm[range(0,nhalo*3,3)]
        y_halos = halo_cm[range(1,nhalo*3,3)]
        z_halos = halo_cm[range(2,nhalo*3,3)]
       
        print('Minimum halo mass:', halomass.min())
        print('Maximum halo mass:', halomass.max())
        
                #halomass_filter=halomass
        logmh=np.log10(halomass)
        logmh=np.array([int(logmh[key]) for key in range(nhalo)])
        
        highmass_filter=np.where(logmh>halo_cutoff_mass_log,logmh,low_mass_log)
       
        #z_min = 0.0
        z_max = nproj*cellsize # See slice() above
    
        mask = z_halos < z_max
        x_halos = x_halos[mask]
        y_halos = y_halos[mask]
        r = highmass_filter[mask]
        r = r/r.max()

        
        s1=plt.scatter(x_halos, y_halos, marker='o', s=30*r, \
                        color='red', alpha=0.9)
        
        
        if(tick_label=='mpc'):
            ax.set_xlim(0,boxsize)
            ax.set_ylim(0,boxsize)
            plt.xlabel('cMpc')
            plt.ylabel('cMpc')
            
        elif(tick_label=='degree'):    
            ax.set_xlim(0,boxsize)
            ax.set_ylim(0,boxsize)
            
            xmin=0
            ymin=0
            xmax=ymax=utils.boxsize_to_degree(halo_redshift, boxsize)
            
            N=4
            xtick_mpc=ytick_mpc=np.linspace(0, boxsize, N)
            
            custom_yticks = np.round(np.linspace(ymin, ymax, N,dtype=float),1)
            
            ax.set_yticks(ytick_mpc)
            ax.set_yticklabels(custom_yticks)
            
            custom_xticks = np.round(np.linspace(xmin, xmax, N,dtype=float),1)
            ax.set_xticks(xtick_mpc)
            ax.set_xticklabels(custom_xticks)
            
            plt.xlabel(r'$X\,(\mathrm{degree})$')
            plt.ylabel('$Y\,(\mathrm{degree})$')
   
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(s, cax=cax)
        cb.set_label(r'$\Delta_\rho$',
                     labelpad=1)
       
        cb.solids.set_edgecolor("face")
        ax.set_aspect('equal', 'box')
        
    if plot_lines:
        """
        Plot a slice of gas density field and overplot the distribution of
        haloes in that slice.
    
        """
        
        with open(dens_gas_file, 'rb') as f:
            dens_gas = np.fromfile(f, dtype='f', count=-1)
            #dens_gas=dens_gas+1.0
            #dens_gas=dens_gas.reshape(ngrid**3,)
            #rhobar = np.mean(dens_gas)
    
    
        
        # Plot gas density 
        i, j, val = slice(dens_gas, ngrid,nproj)
        #val = val/(rhobar*nproj)
    
        cellsize = boxsize/ngrid
        i *= cellsize
        j *= cellsize 
  
        
       
        s = plt.scatter(i, j, c=val, s=10, marker='s',
                       edgecolor='none',
                       cmap=plt.cm.gist_yarg, alpha=0.9)
        
       
        
        halomass, halo_cm=make_halocat(halocat_file,filetype='dat',boxsize=boxsize)
        nhalo=len(halomass)
        # Overplot halos 
        x_halos = halo_cm[range(0,nhalo*3,3)]
        y_halos = halo_cm[range(1,nhalo*3,3)]
        z_halos = halo_cm[range(2,nhalo*3,3)]
       
        print('Minimum halo mass:', halomass.min())
        print('Maximum halo mass:', halomass.max())
        
        #halomass_filter=halomass
        logmh=np.log10(halomass)
        logmh=np.array([int(logmh[key]) for key in range(nhalo)])
        
      
        z_max = nproj*cellsize # See slice() above
    
        mask = z_halos < z_max
        x_halos = x_halos[mask]
        y_halos = y_halos[mask]
        halomass_slice = logmh[mask]
        

   
        mass_cut=halomass_slice >= halo_cutoff_mass_log
        halomass_slice_cut=halomass_slice[mass_cut]
        x_halos_cut= x_halos[mass_cut]
        y_halos_cut= y_halos[mass_cut]
        
        
        
        lot=mhalo_to_lot(halo_redshift, halomass_slice_cut, kind='mean',use_scatter=use_scatter)
        r=halomass_slice_cut/halomass_slice_cut.max()
      
        
        s1=plt.scatter(x_halos_cut, y_halos_cut, marker='o', c=lot, s=50*r,cmap='YlOrRd', vmin=3, vmax=8, alpha=0.9)
        
            
    
        if(tick_label=='mpc'):
            ax.set_xlim(0,boxsize)
            ax.set_ylim(0,boxsize)
            plt.xlabel('cMpc')
            plt.ylabel('cMpc')
            
        elif(tick_label=='degree'):    
            ax.set_xlim(0,boxsize)
            ax.set_ylim(0,boxsize)
            
            xmin=0
            ymin=0
            xmax=ymax=utils.boxsize_to_degree(halo_redshift, boxsize)
            
            N=4
            xtick_mpc=ytick_mpc=np.linspace(0, boxsize, N)
            
            custom_yticks = np.round(np.linspace(ymin, ymax, N,dtype=float),1)
            
            ax.set_yticks(ytick_mpc)
            ax.set_yticklabels(custom_yticks)
            
            custom_xticks = np.round(np.linspace(xmin, xmax, N,dtype=float),1)
            ax.set_xticks(xtick_mpc)
            ax.set_xticklabels(custom_xticks)
            
            plt.xlabel(r'$X\,(\mathrm{degree})$')
            plt.ylabel('$Y\,(\mathrm{degree})$')
    
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(s1, cax=cax)
        cb.set_label(r'$\log(L_{\mathrm{CII}})$',
                     labelpad=1)
       
        cb.solids.set_edgecolor("face")
        ax.set_aspect('equal', 'box')
        
        
        cax1 = divider.append_axes("bottom", "3%", pad="13%")
        cb1 = plt.colorbar(s,cax=cax1,orientation='horizontal')
        cb1.set_label(r'$\Delta$',
                     labelpad=5)
        cb1.solids.set_edgecolor("face")
            
    
    
    plt.tight_layout()
    plt.savefig("slice_plot.pdf",bbox_inches='tight')
  

def plot_beam(theta_fwhm, beam_unit, boxsize, ngrid, nproj, halocat_file, halo_redshift, halo_cutoff_mass_log=11, 
              use_scatter=True, unit='degree', add_noise=False, random_noise_parcentage=None):
    
    
    global final_cov
    xl, yl, lum=calc_luminosity(boxsize, ngrid, nproj,halocat_file, halo_redshift, 
                                halo_cutoff_mass_log=halo_cutoff_mass_log, use_scatter=use_scatter, unit=unit)

    if(beam_unit=='arcmin' or beam_unit=='arcminute' or beam_unit=='minute'):
        print("Theta FWHM (arc-min):", theta_fwhm) 
        theta=theta_fwhm
    if(beam_unit=='second' or beam_unit=='arcsecond' or beam_unit=='arcsec'):
        print("Theta FWHM (arc-second):", theta_fwhm) 
        theta=theta_fwhm/60.0

    luminosity_max=lum.max()
    x_arc=xl*60
    y_arc=yl*60
    
    sx=0.03*(lum**3) #keep it 0.03*(lum**3) 
    sy=0.03*(lum**3)  #keep it 0.03*(lum**3) 
   
    beam_std=theta/(np.sqrt(8*np.log(2.0)))
    gauss_kernel = Gaussian2DKernel(beam_std)
    
    
    x_p=np.linspace(0,x_arc.max(),num=ngrid)
    y_p=np.linspace(0,y_arc.max(),num=ngrid)
    
    x_p,y_p=np.meshgrid(x_p,y_p)
    
    def beam(amp,x,y,sx,sy):
        gauss_b = Gaussian2D(amp,x,y,sx,sy)
        return gauss_b
     
    def beam_conv(amp,x,y,sx,sy, x_p,y_p, kernel=gauss_kernel):
        gauss = Gaussian2D(amp,x,y,sx,sy)
        gauss_data=gauss(x_p,y_p)
        smoothed_data= convolve(gauss_data, gauss_kernel)
        return smoothed_data
    
    flen=len(x_p)
    lum_len=len(lum)
    if(add_noise==False and random_noise_parcentage==None):
        final_conv=0.001*np.ones([flen,flen])
    if(add_noise==True):
        #final_conv=random_noise_parcentage *lum.max()* (np.random.rand(flen, flen)-0.5)
        final_conv=0.001*np.ones([flen,flen])
    for i in range(lum_len):
        b= beam(lum[i],x_arc[i],y_arc[i],sx[i],sy[i])
        gauss_data=b(x_p,y_p)
        if(add_noise==False and random_noise_parcentage==None):
            final_conv=gauss_data+final_conv
        if(add_noise==True):
            final_conv=gauss_data+final_conv+random_noise_parcentage *luminosity_max* (np.random.rand(flen, flen)-0.5)
        final_conv=convolve(final_conv,gauss_kernel)
        
   
    fig, ax = plt.subplots(figsize=(7,7),dpi=100)
 
    res=ax.imshow(final_conv, cmap='gist_heat', interpolation='gaussian',origin='lower', vmin=0.1, vmax=10.0, rasterized=True, alpha=0.9)
    
    
    x_minutes=(60*utils.boxsize_to_degree(halo_redshift, boxsize))
    #ydegree=utils.boxsize_to_degree(halo_redshift, boxsize)
    tick_num=5
    #step=int(x_minutes/tick_num)
    
    ticks=np.linspace(0, x_minutes,num=tick_num)
    
    
    #cell_size=boxsize/float(ngrid)
    cell_size=x_minutes/ngrid
    
    labels = [str(int(xx)) for xx in ticks]
    #locs = [int(xx) for xx in ticks]
    locs = [xx/cell_size for xx in ticks]

    plt.xticks(locs, labels)
    plt.yticks(locs, labels)
    plt.xlabel('arc-min')
    plt.ylabel('arc-min')

    title = '$z={:g}$'.format(halo_redshift)
    plt.title(title, fontsize=18)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "3%", pad="3%")
    cb = plt.colorbar(res, cax=cax)
    cb.set_label(r'$L_{OIII}$', labelpad=20)
    cb.solids.set_edgecolor("face")
    cb.ax.tick_params('both', which='major', length=3, width=1, direction='out')
    plt.savefig("luminsoty_beam.png")
    
    
    
    
