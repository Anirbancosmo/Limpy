#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:43:43 2022

@author: anirbanroy
"""


import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d


def get_fourier_k(Lx, Ly, Lz, Nx, Ny, Nz, fourier_method="half"):
    r"""
    Returns the Fourier frequencies.

    Parameters
    ----------
    Lx, Ly, Lz : float
        Box size in Mpc/h along x, y, z direction.

    Nx, Ny, Nz : int
        Number of grids along x, y, z direction.

    fourier_method : optional
        Do np.fft.rfftfreq for half or np.fft.fftfreq for full.


    Returns
    -------
    kgrid : array
        where :math:`k=\sqrt(k_x^2 + k_y^2 + k_z^2)`

    angle : float
        angle with respect to the z axis.

    """

    # Fourier freequencies along x, y, z axes.

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=(Lx / Nx))
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=(Ly / Ny))

    if fourier_method == "half":
        kz = 2 * np.pi * np.fft.rfftfreq(Nz, d=(Lz / Nz))

    if fourier_method == "full":
        kz = 2 * np.pi * np.fft.fftfreq(Nz, d=(Lz / Nz))

    # Ignore the first elemnt division by zero. The angle should be zero.
    np.seterr(divide="ignore", invalid="ignore")

    # make a k grid
    kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx, ky, kz, indexing="ij")))

    # Calculate angle with respect to z axis
    angle = np.nan_to_num(abs(kz) / kgrid, 0)

    return kgrid, angle


def get_binned_pk(
    X_grid, Lx, Ly, Lz, Nx, Ny, Nz, Y_grid=None, weight_array=None, nbins=20, kbins=None
):
    r"""
    Returns the Fourier frequencies.

    Parameters
    ----------
    X_grid, Y_grid : array
        Data cubes of size [Nx, Ny, Nz]. If X_grid and Y_grid are not the same grid
        then this function calculates the cross power spectrum. If Y_grid is not given,
        then this function calculates auto power spectrum.

    Lx, Ly, Lz : float
        Box size in Mpc/h along x, y, z direction.

    Nx, Ny, Nz : int
        Number of grids along x, y, z direction.

    weight_array : float, array
        Weight should be of the same shape

    nbins: int
        number of bins. If nbins is given, then the kbins will be calculated in between
        fundamental and Nyquiest modes.
    kbins: array
        array of k to calculate binned power spectrum.

    Returns
    -------
    kcen : array
        bin center of k array.

    Pbins : array
        volume normalized power spectrum.

    """
    global power, kgrid
    # Fourier freequencies along x, y, z axes.

    Vbox = Lx * Ly * Lz

    # Vcell= (Lx * Ly * Lz) / (Nx * Ny * Nz)

    if weight_array == None:
        weight_array = np.ones([Nx, Ny, Nz])

    if isinstance(Y_grid, np.ndarray) == True:
        X_grid_fft = np.fft.rfftn(weight_array * X_grid)
        Y_grid_fft = np.fft.rfftn(weight_array * Y_grid)
        power = np.real(X_grid_fft * np.conj(Y_grid_fft))
        del X_grid_fft, Y_grid_fft

    else:
        X_grid_fft = np.fft.rfftn(weight_array * X_grid)
        power = np.real(X_grid_fft * np.conj(X_grid_fft))
        del X_grid_fft
        
    cofactor = (Nx * Ny * Nz) ** 2 / Vbox
    power /= cofactor

    if kbins is None:
        k_F = 2 * np.pi / pow(Vbox, 1 / 3)  # 2 * np.pi /np.max([Lx, Ly])
        k_N = np.pi * Nx / Lx
        delta_k = 2 * k_F
        kbins = np.arange(k_F, k_N, delta_k)

    if kbins is not None:
        kbins = kbins

    kgrid, mu = get_fourier_k(Lx, Ly, Lz, Nx, Ny, Nz)

    Pbins, _, _ = stats.binned_statistic(
        kgrid.flatten(), power.flatten(), statistic="mean", bins=kbins
    )
    del kgrid
    return kbins[:-1], Pbins


def get_pk3d(
    X_grid,
    Lx,
    Ly,
    Lz,
    Nx,
    Ny,
    Nz,
    Y_grid=None,
    weight_array=None,
    nbins=20,
    kbins=None,
):
    r"""
    Returns the Fourier frequencies.

    Parameters
    ----------
    X_grid, Y_grid : array
        Data cubes of size [Nx, Ny, Nz]. If X_grid and Y_grid are not the same grid
        then this function calculates the cross power spectrum. If Y_grid is not given,
        then this function calculates auto power spectrum.

    Lx, Ly, Lz : float
        Box size in Mpc/h along x, y, z direction.

    Nx, Ny, Nz : int
        Number of grids along x, y, z direction.

    weight_array : float, array
        Weight should be of the same shape

    nbins: int
        number of bins. If nbins is given, then the kbins will be calculated in between
        fundamental and Nyquiest modes.
    kbins: array
        array of k to calculate binned power spectrum.

    Returns
    -------
    kcen : array
        bin center of k array.

    Pbins : array
        volume normalized power spectrum.

    """
    global power, kgrid
    # Fourier freequencies along x, y, z axes.

    Vbox = Lx * Ly * Lz

    # Xgrid_sliced=np.array_split(X_grid, Nz_new, axis=2)

    # Vcell= (Lx * Ly * Lz) / (Nx * Ny * Nz)

    if weight_array == None:
        weight_array = np.ones([Nx, Ny, Nz])


    if isinstance(Y_grid, np.ndarray) == True:
        X_grid_fft = np.fft.rfftn(weight_array * X_grid)
        Y_grid_fft = np.fft.rfftn(weight_array * Y_grid)
        power = np.real(X_grid_fft * np.conj(Y_grid_fft))
        del X_grid_fft, Y_grid_fft
    
    else:
        X_grid_fft = np.fft.rfftn(weight_array * X_grid)
        power = np.real(X_grid_fft * np.conj(X_grid_fft))
        del X_grid_fft

    cofactor = (Nx * Ny * Nz) ** 2 / Vbox
    power /= cofactor

    if kbins is None:
        k_F = 2 * np.pi / pow(Vbox, 1 / 3)  # 2 * np.pi /np.max([Lx, Ly])
        k_N = np.pi * Nx / Lx
        delta_k = 2 * k_F
        #delta_k = k_F
        kbins = np.arange(k_F, k_N, delta_k)

    if kbins is not None:
        kbins = kbins

    # kcen = 0.5 * (kbins[1:] + kbins[:-1])

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=(Lx / Nx))
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=(Ly / Ny))

    kz = 2 * np.pi * np.fft.rfftfreq(Nz, d=Lz / Nz)

    # make a k grid
    kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx, ky, kz, indexing="ij")))

    Pbins, _, _ = stats.binned_statistic(
        kgrid.flatten(), power.flatten(), statistic="mean", bins=kbins
    )
    del kgrid

    return kbins[:-1], Pbins


def get_pk2d(X_grid, Lx, Ly, Nx, Ny, Y_grid=None, weight_array=None, kbins=None):
    r"""
    Returns the Fourier frequencies.

    Parameters
    ----------
    X_grid, Y_grid : array
        Data cubes of size [Nx, Ny, Nz]. If X_grid and Y_grid are not the same grid
        then this function calculates the cross power spectrum. If Y_grid is not given,
        then this function calculates auto power spectrum.

    Lx, Ly, Lz : float
        Box size in Mpc/h along x, y, z direction.

    Nx, Ny, Nz : int
        Number of grids along x, y, z direction.

    weight_array : float, array
        Weight should be of the same shape

    nbins: int
        number of bins. If nbins is given, then the kbins will be calculated in between
        fundamental and Nyquiest modes.
    kbins: array
        array of k to calculate binned power spectrum.

    Returns
    -------
    kcen : array
        bin center of k array.

    Pbins : array
        volume normalized power spectrum.

    """
    global power, kgrid
    # Fourier freequencies along x, y, z axes.

    Area = Lx * Ly

    # Xgrid_sliced=np.array_split(X_grid, Nz_new, axis=2)

    # Vcell= (Lx * Ly * Lz) / (Nx * Ny * Nz)

    if weight_array == None:
        weight_array = np.ones([Nx, Ny])


    if isinstance(Y_grid, np.ndarray) == True:
        X_grid_fft = np.fft.rfftn(weight_array * X_grid)
        Y_grid_fft = np.fft.rfftn(weight_array * Y_grid)
    
    else:
        X_grid_fft = np.fft.rfftn(weight_array * X_grid)
        Y_grid_fft = X_grid_fft

    power = np.real(X_grid_fft * np.conj(Y_grid_fft))

    cofactor = (Nx * Ny) ** 2 / Area

    power /= cofactor

    if kbins is None:
        k_F = 2 * np.pi / pow(Area, 1 / 2)  # 2 * np.pi /np.max([Lx, Ly])
        k_N = np.pi * Nx / Lx
        delta_k = 2 * k_F
        kbins = np.arange(k_F, k_N, delta_k)

    if kbins is not None:
        kbins = kbins

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)

    # make a k grid
    kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx, ky, indexing="ij")))

    Pbins, _, _ = stats.binned_statistic(
        kgrid.flatten(), power.flatten(), statistic="mean", bins=kbins
    )

    return kbins[1:], Pbins





def get_pk_mu_nu(X_grid, Lx, Ly, Lz, Nx, Ny, Nz, modes="parallel", line_name="CII", nu_obs=None, dnu=None, Y_grid=None, weight_array=None, nbins=20, kbins=None ):
    r"""
    Returns the Fourier frequencies.

    Parameters
    ----------
    X_grid, Y_grid : array
        Data cubes of size [Nx, Ny, Nz]. If X_grid and Y_grid are not the same grid
        then this function calculates the cross power spectrum. If Y_grid is not given,
        then this function calculates auto power spectrum.
    
    Lx, Ly, Lz : float
        Box size in Mpc/h along x, y, z direction.
        
    Nx, Ny, Nz : int
        Number of grids along x, y, z direction.
        
    weight_array : float, array
        Weight should be of the same shape 
        
    nbins: int
        number of bins. If nbins is given, then the kbins will be calculated in between
        fundamental and Nyquiest modes. 
    kbins: array
        array of k to calculate binned power spectrum. 
        
    Returns
    -------
    kcen : array
        bin center of k array.
    
    Pbins : array
        volume normalized power spectrum. 
    
    """
    global power, kperp,  kgrid_new
    # Fourier freequencies along x, y, z axes.
    
    Vbox =  Lx * Ly * Lz
    
    #Vcell= (Lx * Ly * Lz) / (Nx * Ny * Nz)
    
    if (weight_array==None):
        weight_array=np.ones([Nx, Ny, Nz])
    
    if isinstance(Y_grid, np.ndarray) == True:
        X_grid_fft= np.fft.rfftn(weight_array * X_grid)
        Y_grid_fft = np.fft.rfftn(weight_array * Y_grid)
    else:
        X_grid_fft= np.fft.rfftn(weight_array * X_grid)
        Y_grid_fft = X_grid_fft
        
    power = np.real( X_grid_fft *  np.conj(Y_grid_fft))
    
    
    if kbins is None:
        k_F= np.pi /np.max([Lx, Ly, Lz])
        k_N = np.pi * np.max([Nx, Ny, Nz]) / np.min([Lx, Ly, Lz])
        kbins= np.logspace(np.log10(k_F), np.log10(k_N), nbins)
    
    if kbins is not None:
        kbins = kbins
        
    kcen=0.5*(kbins[1:] + kbins[:-1])
    
    kx=2 * np.pi * np.fft.fftfreq(Nx, d=(Lx/Nx))
    ky=2 * np.pi * np.fft.fftfreq(Ny, d=(Ly/Ny))
    kz=2 * np.pi *  np.fft.rfftfreq(Nz, d=(Lz/Nz))
    
    cofactor= (Nx*Ny*Nz)**2 /Vbox
    power /= cofactor
    

    if modes=="parallel":

        kz=2 * np.pi *  np.fft.rfftfreq(Nz, d=Lz/Nz)
        
        index=getindep(Nx, Ny, Nz)
        kpara_arr = np.tile(kz,(Nx,Ny,1))
        kgrid_new= kpara_arr[index==True]
        power_new= power[index==True]
            
        
    if modes=="perpendicular":        
        index=getindep(Nx, Ny, Nz)
        
        kperp = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx,ky, indexing='ij')))
        kperp_arr = np.reshape(np.repeat(kperp,int(Nz/2)+1) , (Nx,Ny, int(Nz/2)+1))
        
        kgrid_new= kperp_arr[index==True]
        power_new= power[index==True]
    
    Pbins, _, _ = stats.binned_statistic(kgrid_new.flatten(), power_new.flatten(), statistic = "mean", bins = kbins)
    
    return kcen, Pbins



def get_pk_perp_para(
    X_grid,
    Lx,
    Ly,
    Lz,
    Nx,
    Ny,
    Nz,
    line_name="CII",
    nu_obs=None,
    dnu=None,
    Y_grid=None,
    weight_array=None,
    nbins=20,
    k_para_bins=None,
    k_perp_bins=None,
):
    r"""
    Returns the Fourier frequencies.

    Parameters
    ----------
    X_grid, Y_grid : array
        Data cubes of size [Nx, Ny, Nz]. If X_grid and Y_grid are not the same grid
        then this function calculates the cross power spectrum. If Y_grid is not given,
        then this function calculates auto power spectrum.

    Lx, Ly, Lz : float
        Box size in Mpc/h along x, y, z direction.

    Nx, Ny, Nz : int
        Number of grids along x, y, z direction.

    weight_array : float, array
        Weight should be of the same shape

    nbins: int
        number of bins. If nbins is given, then the kbins will be calculated in between
        fundamental and Nyquiest modes.
    kbins: array
        array of k to calculate binned power spectrum.

    Returns
    -------
    kcen : array
        bin center of k array.

    Pbins : array
        volume normalized power spectrum.

    """
    global power, kperp, kgrid_new
    # Fourier freequencies along x, y, z axes.

    Vbox = Lx * Ly * Lz

    if weight_array == None:
        weight_array = np.ones([Nx, Ny, Nz])


    if isinstance(Y_grid, np.ndarray) == True:
        X_grid_fft = np.fft.rfftn(weight_array * X_grid)
        Y_grid_fft = np.fft.rfftn(weight_array * Y_grid)
        
    else:
        X_grid_fft= np.fft.rfftn(weight_array * X_grid)
        Y_grid_fft = X_grid_fft

    power = np.real(X_grid_fft * np.conj(Y_grid_fft))

    if k_perp_bins is None:
        k_F = 2 * np.pi / pow((Lx * Ly), 1 / 2)
        k_N = np.pi * Nx / Lx
        delta_k = 2 * k_F
        k_perp_bins = np.arange(k_F, k_N, delta_k)

    if k_perp_bins is not None:
        k_perp_bins = k_perp_bins

    if k_para_bins is None:
        k_F = 2 * np.pi / Lz
        k_N = np.pi * Nz / Lz
        delta_k = 2 * k_F
        k_para_bins = np.arange(k_F, k_N, delta_k)

    if k_para_bins is not None:
        k_para_bins = k_para_bins

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=(Lx / Nx))
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=(Ly / Ny))
    kz = 2 * np.pi * np.fft.rfftfreq(Nz, d=(Lz / Nz))

    cofactor = (Nx * Ny * Nz) ** 2 / Vbox
    power /= cofactor

    index = getindep(Nx, Ny, Nz)
    kpara_arr = np.tile(kz, (Nx, Ny, 1))

    kgrid_para = kpara_arr[index == True]
    kperp = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx, ky, indexing="ij")))
    kperp_arr = np.reshape(np.repeat(kperp, int(Nz / 2) + 1), (Nx, Ny, int(Nz / 2) + 1))

    kgrid_perp = kperp_arr[index == True]
    power_array = power[index == True]

    result = stats.binned_statistic_2d(
        kgrid_para.flatten(),
        kgrid_perp.flatten(),
        power_array.flatten(),
        statistic="mean",
        bins=[k_para_bins, k_perp_bins],
    )
    Pgrid = result.statistic

    # kgrid=np.meshgrid(k_para_bins , k_perp_bins , indexing='ij')

    # kgrid_cen=np.meshgrid(k_para_cen, k_perp_cen, indexing='ij')

    # dkgrid= np.diff(kgrid)

    # Nmodes= stats.binned_statistic_2d(kgrid_para.flatten(), kgrid_perp.flatten(), kgrid.flatten(), statistic = "sum", bins = [k_para_bins, k_perp_bins] )

    return k_para_bins, k_perp_bins, np.nan_to_num(Pgrid, nan=0.0)


def plot_pk_para_perp(pk_grid, k_para, k_perp, kscale="log", vmin=None, vmax=None):

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    plt.minorticks_on()
    ax.tick_params(which="major", length=4, width=1, direction="out")
    # ax.tick_params(which='minor', length=2, width=1, direction='out')
    ax.minorticks_off()

    # s = plt.imshow(pk_grid, cmap=cm.PuOr, rasterized=True, vmin=vmin, vmax=vmax, norm=colors.LogNorm(), origin="lower")

    if kscale == "linear":
        s = plt.imshow(
            pk_grid,
            cmap=cm.viridis,
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
            norm=colors.LogNorm(),
            origin="lower",
        )

        labels_x_space = np.linspace(10 ** (k_perp.min()), 10 ** k_perp.max(), num=10)
        labels_y_space = np.linspace(10 ** (k_para.min()), 10 ** k_para.max(), num=10)

        xspace = k_perp[1] - k_perp[0]
        yspace = k_para[1] - k_para[0]

        labels_x = [str("{:.2f}".format(x)) for x in labels_x_space]
        locs_x = [x / xspace for x in labels_x_space]

        labels_y = [str("{:.2f}".format(y)) for y in labels_y_space]
        locs_y = [y / yspace for y in labels_y_space]

        # plt.xticks(locs_x, labels_x)
        # plt.yticks(locs_y, labels_y[::-1])
        plt.xlabel("$h$/Mpc")
        plt.ylabel("$h$/Mpc")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(s, cax=cax)
        cb.set_label(r"$P(k)$", labelpad=1)
        cb.solids.set_edgecolor("face")
        cb.ax.tick_params("both", which="major", length=3, width=1, direction="out")
        plt.tight_layout()

    if kscale == "log":
        s = plt.imshow(
            pk_grid,
            cmap=cm.viridis,
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
            norm=colors.LogNorm(),
            origin="lower",
        )

        labels_x_space = np.linspace(10 ** (k_perp.min()), 10 ** k_perp.max(), num=5)
        labels_y_space = np.linspace(10 ** (k_para.min()), 10 ** k_para.max(), num=5)

        kpara_index = interp1d(10**k_para, np.arange(len(k_para)))
        kperp_index = interp1d(10**k_perp, np.arange(len(k_perp)))

        locs_x = kpara_index(labels_x_space)
        locs_y = kperp_index(labels_y_space)

        labels_x = [str("{:.2f}".format(np.log10(x))) for x in labels_x_space]

        labels_y = [str("{:.2f}".format(np.log10(y))) for y in labels_y_space]

        plt.xticks(locs_x, labels_x)
        plt.yticks(locs_y, labels_y)

        plt.xlabel(r"$k_\perp$ [$h$/Mpc]")
        plt.ylabel(r"$k_\parallel$ [$h$/Mpc]")

        plt.xlim(0, len(k_perp))
        plt.ylim(0, len(k_para))

        divider = make_axes_locatable(ax)

        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(s, cax=cax)
        cb.set_label(r"$P(k)$", labelpad=1)
        cb.solids.set_edgecolor("face")
        cb.ax.tick_params("both", which="major", length=3, width=1, direction="out")
        # cb.set_ticks([-10,-8,-6])
        plt.tight_layout()


def getindep(nx, ny, nz):
    indep = np.full((nx, ny, int(nz / 2) + 1), False, dtype=bool)
    indep[:, :, 1 : int(nz / 2)] = True
    indep[1 : int(nx / 2), :, 0] = True
    indep[1 : int(nx / 2), :, int(nz / 2)] = True
    indep[0, 1 : int(ny / 2), 0] = True
    indep[0, 1 : int(ny / 2), int(nz / 2)] = True
    indep[int(nx / 2), 1 : int(ny / 2), 0] = True
    indep[int(nx / 2), 1 : int(ny / 2), int(nz / 2)] = True
    indep[int(nx / 2), 0, 0] = True
    indep[0, int(ny / 2), 0] = True
    indep[int(nx / 2), int(ny / 2), 0] = True
    indep[0, 0, int(nz / 2)] = True
    indep[int(nx / 2), 0, int(nz / 2)] = True
    indep[0, int(ny / 2), int(nz / 2)] = True
    indep[int(nx / 2), int(ny / 2), int(nz / 2)] = True
    return indep
