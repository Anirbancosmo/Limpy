#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import default_params as dp
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def read_sfr(SFR_filename):
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


def sfr_to_lcp_scatter(z, sfr,a_off=dp.default_lcp_scatter_params['a_off'],a_std=dp.default_lcp_scatter_params['a_std'],
                       b_off=dp.default_lcp_scatter_params['b_off'],b_std=dp.default_lcp_scatter_params['b_std']):
    
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

    

def sfr_to_lcp(z, sfr):
    """
    This function returns luminosity of CII lines in the unit of L_sun. This does not include the scatter, rather
    this is the mean relation. 
    
    Values of fitting parameters are taken from Eq (1) from Chung et al. 2020 (arxiv: 1812.08135)
    
    We write two equations as
    \alpha_z=a-b*z
    \beta_z=c-d*z
    values are mentioned 
    """
    a,b,c,d=dp.default_lcp_chung_params['a'],dp.default_lcp_chung_params['b'],\
    dp.default_lcp_chung_params['c'],dp.default_lcp_chung_params['d']
    alpha_z= a-b*z
    beta_z= c- d*z
    log_lcp=alpha_z*np.log10(sfr)+beta_z
    return 10**log_lcp


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
        
    z_sfr, SFR, error_SFR_up, error_SFR_down= read_sfr(sfr_fname)
    
    return z_sfr, SFR, (SFR-error_SFR_down), (SFR+error_SFR_up)
    

def mhalo_to_lcp_unvec(z,logMh, kind='mean'):
    """
    this function returns luminosity of CII lines in the unit of L_sun.
    Kind optitions takes the SFR: mean , up (1-sigma upper bound) and
    down (1-sigma lower bound)
    """
    
    log_lcp_low=dp.default_dummy_values['log_lcp_low']
    if logMh<11:
        return log_lcp_low
    else:
        z_sfr, SFR_mean, SFR_down, SFR_up=mhalo_to_sfr(logMh)
        SFR_mean=np.interp(z,z_sfr,SFR_mean)
        SFR_up=np.interp(z,z_sfr,SFR_up)
        SFR_down=np.interp(z,z_sfr,SFR_down)
       
        if kind=='mean':
            lcp=sfr_to_lcp_scatter(z,SFR_mean)
        if kind=='up':
            lcp =sfr_to_lcp_scatter(z,SFR_up)
        if kind=='down':
            lcp=sfr_to_lcp_scatter(z,SFR_down)
    return lcp


"""
Ugly way to vectorize it...
#TODO: vectorize it by doing optimization 
"""
mhalo_to_lcp = np.vectorize(mhalo_to_lcp_unvec)


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
        z,sfr,err_up, err_down=read_sfr(fname)
        
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
        
       



def plot_slice(boxsize, ngrid, nproj, dens_gas_file, halocat,halo_redshift,halo_cutoff_mass_log=11,
               density_plot=False, halo_overplot=False, plot_lines=False):
    """
    Plot a slice of gas density field and overplot the distribution of
    haloes in that slice.
    
    boxsize: size of box in Mpc
    ngrid: number of grids along the one axis of simulation box
    nproj: cells to project (values could range from 1 to the ngrid)
    halocat: path to halo catalogue file (full path) 
    halo_redshift: redshift of halos
    halo_cutoff_mass_log: cutt off mass of the halos
    density_plot: If true, plot the density distribution
    halo_plot: If True, plot halos
    
    
    """
    global x_halos, y_halos, r_lcp
    
    low_mass_log=0.0
    #z_halos=6.9  #Calculated from the scale factor of halo catalouge
    
    cellsize = boxsize/ngrid
        
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
   
   
    if density_plot:
        # Plot gas density 
        with open(dens_gas_file, 'rb') as f:
            dens_gas = np.fromfile(f, dtype='f', count=-1)
            #dens_gas=dens_gas+1.0
            dens_gas=dens_gas.reshape(ngrid**3,)
            rhobar = np.mean(dens_gas)
        
        
        i, j, val = slice(dens_gas, ngrid,nproj)
        val = val/(rhobar*nproj)
    
        cellsize = boxsize/ngrid
        i *= cellsize
        j *= cellsize 
    
    
        plt.xlabel('cMpc')
        plt.ylabel('cMpc')
    
        s = plt.scatter(i, j, c=val, s=10, marker='s',
                       edgecolor='none', rasterized=True,
                       cmap=plt.cm.gist_yarg, vmax=3.0, vmin=-3.0 )
        
        
        ax.set_xlim(0,boxsize)
        ax.set_ylim(0,boxsize)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(s, cax=cax)
        cb.set_label(r'$\Delta_{\rho}$',
                     labelpad=1)
       
        cb.solids.set_edgecolor("face")
        ax.set_aspect('equal', 'box')
        
        
    
    if halo_overplot:
        
         # Plot gas density 
        with open(dens_gas_file, 'rb') as f:
            dens_gas = np.fromfile(f, dtype='f', count=-1)
            #dens_gas=dens_gas+1.0
            dens_gas=dens_gas.reshape(ngrid**3,)
            rhobar = np.mean(dens_gas)
        
        
        i, j, val = slice(dens_gas, ngrid,nproj)
        val = val/(rhobar*nproj)
    
        cellsize = boxsize/ngrid
        i *= cellsize
        j *= cellsize 
    
    
        plt.xlabel('cMpc')
        plt.ylabel('cMpc')
    
        s = plt.scatter(i, j, c=val, s=10, marker='s',
                       edgecolor='none', rasterized=True,
                       cmap=plt.cm.gist_yarg, vmax=3.0, vmin=-3.0 )
        
       
    
        halomass, halo_cm=make_halocat(halocat,filetype='dat',boxsize=boxsize)
        
        nhalo=len(halomass)
        # Overplot halos 
        x_halos = halo_cm[range(0,nhalo*3,3)]
        y_halos = halo_cm[range(1,nhalo*3,3)]
        z_halos = halo_cm[range(2,nhalo*3,3)]
       
        print 'Minimum halo mass:', halomass.min()
        print 'Maximum halo mass:', halomass.max()
        
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

        
        s1=plt.scatter(x_halos, y_halos, marker='o', s=20*r, \
                        color='C2', alpha=0.9)
        
        ax.set_xlim(0,boxsize)
        ax.set_ylim(0,boxsize)

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
            dens_gas=dens_gas.reshape(ngrid**3,)
            rhobar = np.mean(dens_gas)
    
        
            
        halomass, halo_cm=make_halocat(halocat,filetype='dat',boxsize=boxsize)
        
        nhalo=len(halomass)
        # Overplot halos 
        x_halos = halo_cm[range(0,nhalo*3,3)]
        y_halos = halo_cm[range(1,nhalo*3,3)]
        z_halos = halo_cm[range(2,nhalo*3,3)]
       
        print 'Minimum halo mass:', halomass.min()
        print 'Maximum halo mass:', halomass.max()
        
        #halomass_filter=halomass
        logmh=np.log10(halomass)
        logmh=np.array([int(logmh[key]) for key in range(nhalo)])
        
        highmass_filter=np.where(logmh>halo_cutoff_mass_log,logmh,low_mass_log)
        lcp=mhalo_to_lcp(halo_redshift,highmass_filter, kind='mean')

            
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(1, 1, 1) 
        # Plot gas density 
        i, j, val = slice(dens_gas, ngrid,nproj)
        val = val/(rhobar*nproj)
    
        cellsize = boxsize/ngrid
        i *= cellsize
        j *= cellsize 
    
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
    
        plt.xlabel('cMpc')
        plt.ylabel('cMpc')
    
        s = plt.scatter(i, j, c=val, s=10, marker='s',
                       edgecolor='none', rasterized=True,
                       cmap=plt.cm.gist_yarg, vmax=3.0, vmin=-3.0, alpha=0.2)
   
    
        #z_min = 0.0
        z_max = nproj*cellsize # See slice() above
    
        mask = z_halos < z_max
        x_halos = x_halos[mask]
        y_halos = y_halos[mask]
        r = highmass_filter[mask]
        r = r/r.max()
        
        r_lcp=lcp[mask]
        
        print "r_lcp", r_lcp

        #r_lcp_log=[10**p for p in r_lcp] 
        #plt.scatter(x_halos, y_halos, marker='o', s=50*r_lowmass, rasterized=True,
        #                color='C2', alpha=0.5)
      
        
        s1=plt.scatter(x_halos, y_halos, marker='o', c=r_lcp, s=50*r,cmap='YlOrRd', vmin=4, alpha=0.9,norm=mpl.colors.LogNorm())
    
        ax.set_xlim(0,boxsize)
        ax.set_ylim(0,boxsize)
        
    
    
        #plt.text(5.0,5.0,r'$n_\mathrm{grid}=512^3$')
        #plt.text(5.0,15.0,r'$z=6$')
    
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
        plt.savefig("lines_1_nouv_full_projection1_z7.0.png",bbox_inches='tight')
    
    plt.tight_layout()
    #plt.savefig("slice_plot.pdf",bbox_inches='tight')
  

