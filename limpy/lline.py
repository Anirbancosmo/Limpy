#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import params as p
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import utils  
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.modeling.models import Gaussian2D
from scipy.interpolate import RectBivariateSpline
import matplotlib.colors as colors

import matplotlib as pl
pl.rcParams['xtick.labelsize'] = '10'
pl.rcParams['ytick.labelsize'] = '10'
pl.rcParams['axes.labelsize'] = '15'
pl.rcParams['axes.labelsize'] = '15'


sfr_filepath='../data/'
z,m,sfr_file=np.loadtxt(sfr_filepath+'sfr_beherozzi.dat', unpack=True)

zlen=137 #manually checked 
mlen=int(len(z)/zlen)
zn=z[0:zlen]
mhn=m.reshape(mlen,zlen)[:,0]
sfrn=sfr_file.reshape(mlen,zlen)
sfr_interpolation=RectBivariateSpline(mhn, zn, sfrn)


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



def sfr_to_L_line(z,sfr, line_name='CII', use_scatter=True):
    """
    Calculates lumiosity of the OIII lines from SFR assuming a 3\sigma Gussian scatter. The parameter values for the scattered relation
    is mentioned in defalut_params module. 
    
    Input: z and sfr
    
    return: luminosity of OIII lines in log scale 
    """
    assert (line_name=='OIII' or line_name=='CII' or line_name[0:2]=='CO'), "Not a familiar line."
    
    
    a_off, a_std, b_off, b_std=p.line_scattered_params(line_name)
    
    
    def L_co_log(sfr,alpha, beta):
        nu_co_line=p.nu_rest(line_name)
        L_ir_sun = sfr * 1e10
        L_coprime = (L_ir_sun * 10 **(-beta)) ** (1/alpha)
        L_co = 4.9e-5 * (nu_co_line/ 115.27) ** 3 * L_coprime
        return np.log10(L_co)
    
    
    if np.isscalar(sfr)==True:
        sfr=np.atleast_1d(sfr) ####Convert inputs to arrays with at least one dimension. Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.
    
    sfr_len=len(sfr)
    log_L_line=np.zeros(sfr_len)
    
    if(use_scatter==True):
        if(line_name=='CII' or line_name=='OIII'):
            for i in range(sfr_len):
                a= np.random.normal(a_off,a_std)
                b= np.random.normal(b_off,b_std)
                log_L_line[i]=(a+b*np.log10(sfr[i]))
            
        elif(line_name[0:2]=='CO'):
            for i in range(sfr_len):
                a= np.random.normal(a_off,a_std)
                b= np.random.normal(b_off,b_std)
             
                log_L_line[i]=L_co_log(sfr[i],a, b)
        
                
        return 10**log_L_line
    
    if(use_scatter==False):
        for i in range(sfr_len):
            if(line_name=='CII' or line_name=='OIII'):
                log_L_line[i]=(a_off+b_off*np.log10(sfr[i]))
                
            elif(line_name[0:2]=='CO'):
                log_L_line[i]=L_co_log(sfr[i],a_off, b_off)  
            
        return 10**log_L_line
        

def mhalo_to_sfr(m,z):
    """
    Returns the SFR history for discrete values of halo mass.
    """
    
    res=sfr_interpolation(m,z) 
    
    res=np.where(res<1e-4, p.lcp_low, res)
    return res.flatten()


def mhalo_to_lline(Mh, z, line_name='CII',use_scatter=False):
    """
    This function returns luminosity of lines (following the input line_name) in the unit of L_sun.
    Kind optitions takes the SFR: mean , up (1-sigma upper bound) and
    down (1-sigma lower bound)
    """
    #if
    
    sfr=mhalo_to_sfr(Mh,z)
    L_line=sfr_to_L_line(z,sfr, line_name=line_name, use_scatter=use_scatter)
    
    return L_line


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


def plot_sfr_mhalo(Mhalo_in,colorlist=None,figname=None):
    
    colorlist=['Crimson','darkgreen','C1','blue','C5','C9']
    lw_def=3.0
    font_legend_def=12
    #alpha_def=0.3
    
    x_min_def=0.1
    x_max_def=8.0
    y_min_def=1e-5
    y_max_def=10000

    
    for i in range (len(Mhalo_in)):
        Mhalo=Mhalo_in[i]
        
        if(Mhalo>=1e9 and Mhalo<1e11):
            zmax=8.0
        
        if(Mhalo>=1e11 and Mhalo<=1e12):
            zmax=5.7
        
        if(Mhalo>1e12 and Mhalo<=1e13):
            zmax=3.0
            
        if(Mhalo>1e13 and Mhalo<=1e14):
            zmax=1.8
        
        if(Mhalo>1e14 and Mhalo<=1e15):
            zmax=0.5
        if(Mhalo>=1e15):
            zmax=0.4
            
        z=np.linspace(0.01, zmax)
        sfr=mhalo_to_sfr(Mhalo,z)
        #plot
        if colorlist:
            plt.plot(z,sfr,lw=lw_def,color=colorlist[i],label=r"$M_{halo}=10^{%d}\,M_\odot$" %np.log10(Mhalo))
            #plt.fill_between(z, sfr-err_down, sfr+err_up,color=colorlist[i],label='',alpha=alpha_def)
        else:
            plt.plot(z,sfr,lw=lw_def,color=colorlist[i],label=r"$M_{halo}=10^{%d}\,M_\odot$" %np.log10(Mhalo))
            #plt.fill_between(z, sfr-err_down, sfr+err_up,color=colorlist[i],label='',alpha=alpha_def)
      

    plt.yscale("log")
    plt.xlim(x_min_def,x_max_def)
    plt.ylim(y_min_def,y_max_def)
    
    plt.ylabel(r"$\mathrm{SFR}\,\,(M_\odot/\mathrm{yr})$")
    plt.xlabel(r'$z$')
    
    plt.legend(loc=1, ncol=2, frameon=False, fontsize=font_legend_def)
        
    if figname:
        plt.savefig("%s" %(figname))
    else:
        plt.savefig("z_sfr_mhalo.png")
        
       
def save_luminosity_slice(boxsize, ngrid, nproj,halocat_file,halo_redshift,line_name='CII',halo_cutoff_mass=1e11, use_scatter=True,save_unit='degree',saved_file_name=None):
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
    #logmh=np.log10(halomass)
    #logmh=np.array([int(logmh[key]) for key in range(nhalo)])
    
  
    z_max = nproj*cellsize # See slice() above
    
    mask = z_halos < z_max
    x_halos = x_halos[mask]
    y_halos = y_halos[mask]
    halomass_slice = halomass[mask]
    

   
    mass_cut=halomass_slice >= halo_cutoff_mass
    halomass_slice_cut=halomass_slice[mass_cut]
    x_halos_cut= x_halos[mass_cut]
    y_halos_cut= y_halos[mass_cut]
    
    lcp=mhalo_to_lline(halomass_slice_cut, halo_redshift, line_name=line_name, use_scatter=use_scatter)
    
    xdegree=utils.physical_boxsize_to_degree(halo_redshift,x_halos_cut )
    ydegree=utils.physical_boxsize_to_degree(halo_redshift,y_halos_cut)
    
    if(saved_file_name==None):
        fname=("luminosity_CII_nproj_%d_z%1.2f" %(nproj, halo_redshift))
    else:
        fname=saved_file_name
    
    if(save_unit=='mpc'):
        np.savez(fname,x=x_halos_cut,y=y_halos_cut, luminosity=lcp)
    if(save_unit=='degree'):
        np.savez(fname,x=xdegree,y=ydegree, luminosity=lcp)


       
def calc_luminosity(boxsize, ngrid, nproj,halocat_file,halo_redshift, line_name='CII',halo_cutoff_mass=1e11, use_scatter=False,halocat_file_type='npz', unit='degree'):
    '''
    Calculate luminosity for input parameters
    '''
  
    #low_mass_log=0.0
    cellsize = boxsize/ngrid
    
    halomass, halo_cm=make_halocat(halocat_file,filetype=halocat_file_type,boxsize=boxsize)
    
    nhalo=len(halomass)
    # Overplot halos 
    x_halos = halo_cm[range(0,nhalo*3,3)]
    y_halos = halo_cm[range(1,nhalo*3,3)]
    z_halos = halo_cm[range(2,nhalo*3,3)]
   
    print('Minimum halo mass:', halomass.min())
    print('Maximum halo mass:', halomass.max())
   
    #halomass_filter=halomass
    #mh=halomass)
    #logmh=np.array([int(logmh[key]) for key in range(nhalo)])
    
  
    z_max = nproj*cellsize # See slice() above

    mask = z_halos < z_max
    x_halos = x_halos[mask]
    y_halos = y_halos[mask]
    halomass_slice = halomass[mask]
        
    mass_cut=halomass_slice >= halo_cutoff_mass
    halomass_slice_cut=halomass_slice[mass_cut]
    x_halos_cut= x_halos[mass_cut]
    y_halos_cut= y_halos[mass_cut]
    
    hcut_len=len(halomass_slice_cut)
    lcp=np.zeros(hcut_len)
    for i in range(hcut_len):
        lcp[i]=mhalo_to_lline(halomass_slice_cut[i],halo_redshift,line_name=line_name, use_scatter=use_scatter)
   
    xdegree=utils.comoving_boxsize_to_degree(halo_redshift,x_halos_cut )
    ydegree=utils.comoving_boxsize_to_degree(halo_redshift,y_halos_cut)
    
    if(unit=='mpc'):
        return x_halos_cut, y_halos_cut, lcp
    if(unit=='degree'):
        return xdegree, ydegree, lcp



    
def plot_slice(boxsize, ngrid, nproj, dens_gas_file, halocat_file,halo_redshift,line_name='CII', halocat_file_type='dat',halo_cutoff_mass=1e11, use_scatter=True,
               density_plot=False, halo_overplot=False, plot_lines=False, unit='mpc'):
    
    

    """
    Plot a slice of gas density field and overplot the distribution of
    haloes in that slice.
    
    boxsize: size of box in Mpc
    ngrid: number of grids along the one axis of simulation box
    nproj: cells to project (values could range from 1 to the ngrid)
    halocat_file: path to halo catalogue file (full path) 
    halo_redshift: redshift of halos
    halo_cutoff_mass: cutt off mass of the halos
    density_plot: If true, plot the density distribution
    halo_plot: If True, plot halos
    
    tick_label=either 'mpc' or degree
    
    """
    
    low_mass_log=0.0
    #z_halos=6.9  #Calculated from the scale factor of halo catalouge
    
    cellsize = boxsize/ngrid
        
    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
   
   
    if density_plot:
        
                # Load density file
        with open(dens_gas_file, 'rb') as f:
            dens_gas = np.fromfile(f, dtype='f', count=-1)
            #dens_gas=dens_gas+1.0
            #dens_gas=dens_gas.reshape(ngrid,ngrid,ngrid)
            #rhobar = np.mean(dens_gas)
        
        # slice the data cube
        i, j, val = slice(dens_gas, ngrid,nproj)
        
        dens_mean=np.mean(dens_gas+1)
        
        val= (val/(dens_mean*nproj))
    
    
        cellsize = boxsize/ngrid
        i *= cellsize
        j *= cellsize 
               

  
        s = plt.scatter(i, j, c=val, s=10, marker='s',
                       edgecolor='none', rasterized=True,
                       cmap=plt.cm.viridis_r, vmax=1, vmin=-1)
        
        
                
        if(unit=='mpc'):
            ax.set_xlim(0,boxsize)
            ax.set_ylim(0,boxsize)
            plt.xlabel('cMpc')
            plt.ylabel('cMpc')
            
        elif(unit=='degree'):    
            ax.set_xlim(0,boxsize)
            ax.set_ylim(0,boxsize)
            
            xmin=0
            ymin=0
            xmax=ymax=utils.comoving_boxsize_to_degree(halo_redshift, boxsize)
            
            N=4
            xtick_mpc=ytick_mpc=np.linspace(0, boxsize, N)
            
            custom_yticks = np.round(np.linspace(ymin, ymax, N,dtype=float),1)
            
            ax.set_yticks(ytick_mpc)
            ax.set_yticklabels(custom_yticks)
            
            custom_xticks = np.round(np.linspace(xmin, xmax, N,dtype=float),1)
            ax.set_xticks(xtick_mpc)
            ax.set_xticklabels(custom_xticks)
            
            plt.xlabel(r'$\Theta\,(\mathrm{degree})$')
            plt.ylabel(r'$\Theta\,(\mathrm{degree})$')
   
        divider = make_axes_locatable(ax)
       
        cax = divider.append_axes("bottom", "3%", pad="13%")
        cb = plt.colorbar(s,cax=cax,orientation='horizontal')
        cb.set_label(r'$\Delta_\rho$',
                     labelpad=5)
       
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
        
        dens_mean=np.mean(dens_gas+1)
        
        val= (val/(dens_mean*nproj))
    
    
        cellsize = boxsize/ngrid
        i *= cellsize
        j *= cellsize 
               

  
        s = plt.scatter(i, j, c=val, s=10, marker='s',
                       edgecolor='none', rasterized=True,
                       cmap=plt.cm.viridis_r, vmax=1, vmin=-1)
        
   
    
        halomass, halo_cm=make_halocat(halocat_file,filetype='dat',boxsize=boxsize)
        
        nhalo=len(halomass)
        # Overplot halos 
        x_halos = halo_cm[range(0,nhalo*3,3)]
        y_halos = halo_cm[range(1,nhalo*3,3)]
        z_halos = halo_cm[range(2,nhalo*3,3)]
       
        print('Minimum halo mass:', halomass.min())
        print('Maximum halo mass:', halomass.max())
        
                #halomass_filter=halomass
        #logmh=np.log10(halomass)
        #logmh=np.array([int(logmh[key]) for key in range(nhalo)])
        
        #highmass_filter=np.where(logmh>halo_cutoff_mass,logmh,low_mass_log)
        
        highmass_filter=np.where(halomass>halo_cutoff_mass,halomass,low_mass_log)
        
       
        #z_min = 0.0
        z_max = nproj*cellsize # See slice() above
    
        mask = z_halos < z_max
        x_halos = x_halos[mask]
        y_halos = y_halos[mask]
        r = highmass_filter[mask]
        r = r/r.max()

        
        s1=plt.scatter(x_halos, y_halos, marker='o', s=100*r, \
                        color='red', alpha=0.9)
        
        
        if(unit=='mpc'):
            ax.set_xlim(0,boxsize)
            ax.set_ylim(0,boxsize)
            plt.xlabel('cMpc')
            plt.ylabel('cMpc')
            
        elif(unit=='degree'):    
            ax.set_xlim(0,boxsize)
            ax.set_ylim(0,boxsize)
            
            xmin=0
            ymin=0
            xmax=ymax=utils.comoving_boxsize_to_degree(halo_redshift, boxsize)
            
            N=4
            xtick_mpc=ytick_mpc=np.linspace(0, boxsize, N)
            
            custom_yticks = np.round(np.linspace(ymin, ymax, N,dtype=float),1)
            
            ax.set_yticks(ytick_mpc)
            ax.set_yticklabels(custom_yticks)
            
            custom_xticks = np.round(np.linspace(xmin, xmax, N,dtype=float),1)
            ax.set_xticks(xtick_mpc)
            ax.set_xticklabels(custom_xticks)
            
            plt.xlabel(r'$\Theta\,(\mathrm{degree})$')
            plt.ylabel(r'$\Theta\,(\mathrm{degree})$')
   
        divider = make_axes_locatable(ax)
       
        cax = divider.append_axes("bottom", "3%", pad="13%")
        cb = plt.colorbar(s,cax=cax,orientation='horizontal')
        cb.set_label(r'$\Delta_\rho$',
                     labelpad=5)
       
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
        
        dens_mean=np.mean(dens_gas+1)
        
        val= (val/(dens_mean*nproj))
        
        
        s = plt.scatter(i, j, c=val, s=10, marker='s',
                       edgecolor='none', rasterized=False,
                       cmap='viridis_r',vmax=1, vmin=-1, alpha=0.9)
        
       

    
        xl, yl, lum=calc_luminosity(boxsize, ngrid, nproj,halocat_file, halo_redshift, line_name=line_name,
                                halo_cutoff_mass=halo_cutoff_mass, halocat_file_type=halocat_file_type, use_scatter=use_scatter, unit='mpc')
    

        
        r=(np.log10(lum)/np.log10(lum.max()))**6
     
        
        s1=plt.scatter(xl, yl, marker='o', c=lum, s=70*r,cmap='afmhot',  vmin=1e4, vmax=1e10, norm=colors.LogNorm(), alpha=0.9)
        
    
        if(unit=='mpc'):
            ax.set_xlim(0,boxsize)
            ax.set_ylim(0,boxsize)
            plt.xlabel('cMpc')
            plt.ylabel('cMpc')
            
        elif(unit=='degree'):    
            ax.set_xlim(0,boxsize)
            ax.set_ylim(0,boxsize)
            
            xmin=0
            ymin=0
            xmax=ymax=utils.comoving_boxsize_to_degree(halo_redshift, boxsize)
            
            N=4
            xtick_mpc=ytick_mpc=np.linspace(0, boxsize, N)
            
            custom_yticks = np.round(np.linspace(ymin, ymax, N,dtype=float),1)
            
            ax.set_yticks(ytick_mpc)
            ax.set_yticklabels(custom_yticks)
            
            custom_xticks = np.round(np.linspace(xmin, xmax, N,dtype=float),1)
            ax.set_xticks(xtick_mpc)
            ax.set_xticklabels(custom_xticks)
            
            plt.xlabel(r'$\Theta\,(\mathrm{degree})$')
            plt.ylabel(r'$\Theta\,(\mathrm{degree})$')
    
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(s1, cax=cax)
        cb.set_label(r'$L_{\mathrm{%s}}\,[L_\odot]$' %(line_name),
                     labelpad=1)
       
        cb.solids.set_edgecolor("face")
        ax.set_aspect('equal', 'box')
        
        
        cax1 = divider.append_axes("bottom", "3%", pad="13%")
        cb1 = plt.colorbar(s,cax=cax1,orientation='horizontal')
        cb1.set_label(r'$\Delta_\rho$',
                     labelpad=5)
        cb1.solids.set_edgecolor("face")
        
    
    plt.tight_layout()
    plt.savefig("slice_plot.pdf",bbox_inches='tight')
  


def plot_beam(theta_fwhm, beam_unit, boxsize, ngrid, nproj, halocat_file, halo_redshift, line_name='CII', halo_cutoff_mass=1e11, 
              use_scatter=True, halocat_file_type='npz', unit='degree', plot_unit='minute', tick_num=5, add_noise=False, random_noise_parcentage=None):
    
    
    global final_cov
    #mtd=p.default_constants['minute_to_degree']
    dtm=p.degree_to_minute
    
    xl, yl, lum=calc_luminosity(boxsize, ngrid, nproj,halocat_file, halo_redshift, line_name=line_name,
                                halo_cutoff_mass=halo_cutoff_mass, halocat_file_type= halocat_file_type, use_scatter=use_scatter, unit=unit)

    if(beam_unit=='arcmin' or beam_unit=='arcminute' or beam_unit=='minute'):
        print("Theta FWHM (arc-min):", theta_fwhm) 
        theta=theta_fwhm
    if(beam_unit=='second' or beam_unit=='arcsecond' or beam_unit=='arcsec'):
        print("Theta FWHM (arc-second):", theta_fwhm) 
        theta=theta_fwhm/60.0

    luminosity_max=lum.max()
    x_arc=xl*dtm
    y_arc=yl*dtm
    
    #sx=2e-2*np.log10(lum)**2 #keep it 0.03*(lum**3) 
    #sy=2e-2*np.log10(lum)**2 #keep it 0.03*(lum**3) 
    sx=np.zeros(len(lum))
    sy=np.zeros(len(lum))
   
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
        final_conv=0.01*np.ones([flen,flen])
    for i in range(lum_len):
        b= beam(lum[i],x_arc[i],y_arc[i],sx[i],sy[i])
        gauss_data=b(x_p,y_p)
        if(add_noise==False and random_noise_parcentage==None):
            final_conv=gauss_data+final_conv
        if(add_noise==True):
            final_conv=gauss_data+final_conv+random_noise_parcentage *luminosity_max* (np.random.rand(flen, flen)-0.5)
        final_conv=convolve(final_conv,gauss_kernel)
        
   
    fig, ax = plt.subplots(figsize=(8,8),dpi=70)
 
    res=ax.imshow(final_conv, cmap='gist_heat', interpolation='gaussian',origin='lower', rasterized=True, alpha=0.9, vmin=1e3, vmax=1e9, norm=colors.LogNorm())
    
        
    if(plot_unit=='degree'):
        x_tick=(utils.comoving_boxsize_to_degree(halo_redshift, boxsize))
        cell_size=x_tick/ngrid
        ticks=np.linspace(0, x_tick,num=tick_num)
        labels = [str("{:.1e}".format(xx)) for xx in ticks]
        locs = [xx/cell_size for xx in ticks]
        plt.xlabel(r'$\Theta\,(\mathrm{degree})$')
        plt.ylabel(r'$\Theta\,(\mathrm{degree})$')

        
    if(plot_unit=='minute'):
        x_tick=(dtm*utils.comoving_boxsize_to_degree(halo_redshift, boxsize))
        cell_size=x_tick/ngrid
        ticks=np.linspace(0, x_tick,num=tick_num)
        labels = [str("{:.1f}".format(xx)) for xx in ticks]
        locs = [xx/cell_size for xx in ticks]
        plt.xlabel(r'$\Theta\,(\mathrm{arc-min})$')
        plt.ylabel(r'$\Theta\,(\mathrm{arc-min})$')

    

    plt.xticks(locs, labels)
    plt.yticks(locs, labels)
    
    title = '$z={:g}$'.format(halo_redshift)
    plt.title(title, fontsize=18)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "3%", pad="3%")
    cb = plt.colorbar(res, cax=cax)
    cb.set_label(r'$L$', labelpad=20)
    cb.solids.set_edgecolor("face")
    cb.ax.tick_params('both', which='major', length=3, width=1, direction='out')
    
    plt.tight_layout()
    plt.savefig("luminsoty_beam.png")
    
