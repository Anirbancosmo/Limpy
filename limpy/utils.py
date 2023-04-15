#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:57:35 2020

@author: anirbanroy
"""
import imp

import numpy as np
from scipy.integrate import simps

import limpy.cosmos as cosmos
import limpy.params as p

imp.reload(cosmos)
imp.reload(p)




def volume_box(boxsize):
    """
    Volume of the simulation box.
    
    Parameters
    ----------
    boxsize: float
        the length of the box in MPc/h
    
    Returns
    -------
        volume in (Mpc/h)^3
    """
    
    return boxsize**3


def volume_cell(boxsize, ngrid):
    """
    Volume of the cells (for a cube).
    
    Parameters
    ----------
    boxsize: float
        the length of the box in MPc/h
        
    ngrid: int
        Number of grids along the one direction of box.
    
    Returns
    -------
        volume in (Mpc/h)^3
    """
    
    clen = boxsize / ngrid  # length of a cell
    return clen**3  # in (Mpc/h)**3


def angle_to_comoving_size(z, angle):
    """
    get the comoving size for an angle at redshift z.
    
    Parameters
    ----------
    z: float
        redshift
    angle: float
        angle in radian.
            
    
    Returns
    -------
        size in (Mpc/h)
    """

    dc = p.cosmo.D_co(z)
    size = angle * dc
    return size


def comoving_boxsize_to_angle(z, boxsize):
    """
    Angle substended by the surface of a box at redshift z.
    
    Parameters
    ----------
    boxsize: float
        the length of the box in MPc/h
        
    z: float
        redshift
    
            
    
    Returns
    -------
    Angle: float
        angle in radian unit.
    """

    da = p.cosmo.D_co(z)
    theta_rad = boxsize / da
    return theta_rad


def angle_to_comoving_boxsize(z, angle, angle_unit="degree"):
    """
    Angle substended by the surface of a box at redshift z.
    
    Parameters
    ----------
    z: float
        redshift
    
    angle:
        The angle in radian or degree defined by angle_unit.
    
    angle_unit: str
        The unit of angle, either in degree or radian.
    
    Returns
    -------
    boxsize: float
        The comoving boxsize in (Mpc/h)
    """
    
    if angle_unit == "degree":
        theta_rad = angle * np.pi / 180

    if angle_unit == "radian":
        theta_rad = angle

    da = p.cosmo.D_co(z)
    boxsize = theta_rad * da
    return boxsize


def physical_boxsize_to_angle(z, boxsize):    
    da = p.cosmo.D_angular(z)
    theta_rad = boxsize / da
    return theta_rad


def length_projection(z=None, dz=None, nu_obs=None, dnu=None, line_name="CII"):
    """
    This function returns the projection length for the frequency resolution, dnu_obs.
    
    Parameters
    ----------
    z: float
        redshift
    
    dz: float
        redshift bin size. z and dz have to be passed toegther. 
    
    nu_obs: float
        observational frequency in GHz.
    
    dnu_obs: float
        observational frequency resolution in GHz.   
    
    line_name: str
        the name of the lines. Check "line_list" to get all the available lines.
    
    angle_unit: str
        The unit of angle, either in degree or radian.
    
    Returns
    -------
    boxsize: float
        The comoving boxsize in (Mpc/h)
    """
    
    if (z is None and dz is not None) or (z is not None and dz is None):
        raise ValueError(
            "Specify z and dz together to calculate the projection length calculation."
        )

    if (nu_obs is None and dnu is not None) or (nu_obs is not None and dnu is None):
        raise ValueError(
            "Specify nu_obs and dnu together to calculate the projection length calculation."
        )

    if (z is None and dz is None) and (nu_obs is None and dz is None):
        raise ValueError(
            "Either specify z and dz together or nu_obs and dnu together for projection length calculation."
        )

    if z != None and dz != None:
        dco1 = p.cosmo.D_co(z)
        dco2 = p.cosmo.D_co(z + dz)
        res = dco2 - dco1

    if nu_obs != None and dnu != None:
        z_obs1 = nu_obs_to_z(nu_obs, line_name=line_name)
        z_obs2 = nu_obs_to_z((nu_obs + dnu), line_name=line_name)

        dco1 = p.cosmo.D_co(z_obs1)
        dco2 = p.cosmo.D_co(z_obs2)
        res = dco1 - dco2

    return res



def convert_beam_unit_to_radian(theta_beam, beam_unit):
    """
    Converts the beam in radian
    
    Parameters
    ----------
    theta_beam: float
        FWHM of the beam
        
    beam_unit: str
        The unit of angle, either in arcmin, degree or radian.
    
    Returns
    -------
    boxsize: float
        The comoving boxsize in (Mpc/h)
    """
    
    if beam_unit == "arcmin" or beam_unit == "min" or beam_unit == "minute":
        theta_beam *= (1.0 / 60) * (np.pi / 180)
    if beam_unit == "arcsec" or beam_unit == "sec" or beam_unit == "second":
        theta_beam *= (1.0 / 3600) * (np.pi / 180)
    if beam_unit == "degree" or beam_unit == "deg":
        theta_beam *= np.pi / 180
    if beam_unit == "radian" or beam_unit == "rad":
        theta_beam = theta_beam

    return theta_beam


def sigma_beam(theta_beam, beam_unit="arcmin"):
    """
    Parameters
    ----------
    theta_beam: float
        FWHM of the beam
        
    beam_unit: str
        The unit of angle, either in arcmin, degree or radian.
    
    Returns
    -------
    the standard deviation of the beam.
    """
    
    theta = convert_beam_unit_to_radian(theta_beam, beam_unit=beam_unit)
    return theta / np.sqrt(8 * np.log(2))


def Omega_beam(theta_beam, beam_unit="arcmin"):
    """
    Parameters
    ----------
    theta_beam: float
        FWHM of the beam
        
    beam_unit: str
        The unit of angle, either in arcmin, degree or radian.
    
    Returns
    -------
    the standard deviation of the beam.
    """
    
    theta_rad = convert_beam_unit_to_radian(theta_beam, beam_unit=beam_unit)
    return np.pi * theta_rad**2 / 4 * np.log(2)


def sigma_beam_parallel(theta_beam, z, beam_unit="arcmin"):
    beam_par = p.cosmo.D_co(z) * sigma_beam(theta_beam, beam_unit=beam_unit)
    return beam_par  # unit in mpc/h


def sigma_beam_perpendicular(z, nu_obs, delta_nu):
    res = (p.c_in_mpc / p.cosmo.H_z(z)) * ((1 + z) * delta_nu / nu_obs)
    return res * p.small_h  # unit in mpc/h


def W_beam_Li(k, theta_beam, z, nu_obs, delta_nu, beam_unit="arcmin"):
    sigma_para = sigma_beam_parallel(theta_beam, z, beam_unit=beam_unit)
    sigma_perp = sigma_beam_perpendicular(z, nu_obs, delta_nu)

    mu = np.linspace(0, 1, num=200)
    sigma_sq = -(k**2) * (sigma_para**2 - sigma_perp**2)

    if sigma_sq >= 400.0:
        sigma_sq = 400.0
    else:
        sigma_sq = sigma_sq

    k1 = -(k**2) * sigma_perp**2

    integrand1 = np.exp(k1)

    integrand = np.exp(sigma_sq * mu**2)

    print(integrand)
    res = integrand1 * simps(integrand, mu)

    return res




def t_pix(theta_beam, tobs_total, Ndet_eff, S_area, beam_unit="arcmin"):
    """
    Time per pixel.

    theta_min: the beam size in arc-min.
    tobs_total: total observing time.
    Ndet_eff: Effective number of detectors, for CCATp, Ndet_eff~20.
    S_area: Survey area in degree^2.

    return: t_pix in second.

    """
    omega_beam = Omega_beam(theta_beam, beam_unit=beam_unit)
    S_area_rad = S_area * (p.degree_to_radian) ** 2

    tobs_total *= 3600  # hours to seconds
    res = tobs_total * Ndet_eff * omega_beam / (S_area_rad)
    return res


def V_surv(z, S_area, B_nu, line_name="CII158"):
    """
    Calculates the survey volume in MPc.

    z: redshift
    lambda_line: frequncy of line emission in micrometer
    A_s: Survey area in degree**2
    B_nu: Total frequency band width resolution in GHz

    return: Survey volume.
    """

    nu = p.nu_rest(line_name)
    Sa_rad = S_area * (p.degree_to_radian) ** 2

    lambda_line = p.freq_to_lambda(nu)  # mpc/h

    y = lambda_line * (1 + z) ** 2 / p.cosmo.H_z(z)
    res = p.cosmo.D_co(z) ** 2 * y * (Sa_rad) * B_nu
    return res  # (Mpc/h)^3


def nu_obs_to_z(nu_obs, line_name="CII158"):
    """
    This function evaluates the redshift of a particular line emission
    corresponding to the observed frequency.

    return: redshift of line emission.
    """

    global nu_rest_line

    nu_rest_line = p.nu_rest(line_name=line_name)

    if nu_obs >= nu_rest_line:
        z = 0

    else:
        z = (nu_rest_line / nu_obs) - 1
    return z


def comoving_size_to_delta_nu(length, z, line_name="CII158"):
    nu_rest_line = p.nu_rest(line_name=line_name)

    dchi_dz = p.c_in_mpc / p.cosmo.H_z(z)

    dnu = (length) * nu_rest_line / (dchi_dz * (1 + z) ** 2)

    return dnu


def solid_angle(length, z):

    "Solid angle in Sr unit"
    A = length * length
    Dco = p.cosmo.D_co(z)
    return A / Dco**2


def V_pix(z, theta_beam, delta_nu, beam_unit="arcmin", line_name="CII158"):
    """
    z: redshift
    lambda_line: frequncy of line emission in micrometer
    theta_min: beam size in arc-min
    delta_nu: the frequency resolution in GHz
    """

    theta_rad = convert_beam_unit_to_radian(theta_beam, beam_unit=beam_unit)

    nu = p.nu_rest(line_name)

    lambda_line = p.freq_to_lambda(nu)

    y = lambda_line * (1 + z) ** 2 / p.cosmo.H_z(z)
    res = p.cosmo.D_co(z) ** 2 * y * (theta_rad) ** 2 * delta_nu
    return res  # (Mpc/h)^3


def box_freq_to_quantities(
    nu_obs=280, dnu_obs=2.8, boxsize=80, ngrid=512, z_start=None, line_name="CII"
):

    nu_rest = p.nu_rest(line_name=line_name)

    cell_size = boxsize / ngrid

    if z_start:
        z_em = z_start
    else:
        z_em = (nu_rest / nu_obs) - 1

    dz_em = nu_rest * dnu_obs / (nu_obs * (nu_obs + dnu_obs))
    d_chi = p.cosmo.D_co(z_em + dz_em) - p.cosmo.D_co(z_em)
    d_ngrid = int(d_chi / cell_size)

    return round(z_em, 2), round(dz_em, 2), d_chi, d_ngrid


def get_lines_same_frequency(line_list, nu_obs=220, dnu_obs=40, zlim=15):
    global z_em
    list_len = len(line_list)
    z_em =  np.zeros(list_len)
    dz_em = np.zeros(list_len)
    
    for i in range(list_len):
        nu_rest = p.nu_rest(line_name=line_list[i]) 
        z_em[i] = (nu_rest / nu_obs) - 1
        dz_em[i] = nu_rest * dnu_obs / (nu_obs * (nu_obs + dnu_obs))
        
        
            
    mask = (z_em>0) & (z_em<zlim)
    z_em= z_em[mask]
    dz_em= dz_em[mask]
    line_names= np.array(line_list)[mask]   
    
    return np.round(z_em, 2), dz_em,  line_names



def sigma_noise(theta_min, NEI, beam_unit="arcmin"):
    """
    noise per pixel.
    Eq 22 and 23 of https://arxiv.org/pdf/1802.04804.pdf.
    """

    # omegab=Omega_beam(theta_min, beam_unit=beam_unit)
    return NEI  # * 1e-9/omegab


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

    Pn = (
        V_pix(z, theta_min, delta_nu)
        * sigma_noise(theta_min, NEI) ** 2
        / (t_pix(theta_min, tobs_total, Nspec_eff, S_a))
    )
    return Pn


def P_noise_ccatp(nu=220):
    if nu == 220:
        res = 2.6e9

    elif nu == 280:
        res = 4.9e9

    elif nu == 350:
        res = 3.9e10

    elif nu == 410:
        res = 1.2e11

    return res


def N_modes(k, z, delta_k, A_s, B_nu, line_name="CII"):
    Vs = V_surv(z, A_s, B_nu, line_name=line_name)
    res = 2 * np.pi * k**2 * delta_k * Vs / (2 * np.pi) ** 3
    return res


def slice(datacube, ngrid, nproj, option="C"):
    """
    Produces a slice from a 3D data cube for plotting. `option' controls
    whether data cube has C or Fortran ordering.

    """
    datacube = datacube.reshape(ngrid**3)

    iarr = np.zeros(ngrid * ngrid)
    jarr = np.zeros(ngrid * ngrid)
    valarr = np.zeros(ngrid * ngrid)

    counter = 0
    for i in range(ngrid):
        for j in range(ngrid):
            iarr[counter] = i
            jarr[counter] = j
            valarr[counter] = 0.0
            for k in range(nproj):
                if option == "F":
                    valarr[counter] += datacube[i + ngrid * (j + ngrid * k)]
                elif option == "C":
                    valarr[counter] += datacube[k + ngrid * (j + ngrid * i)]
            counter += 1

    return iarr, jarr, valarr


def freq_2D(boxsize, ngrid):
    kf = 2.0 * np.pi / boxsize
    kn = np.pi / (boxsize / ngrid)

    return kf, kn


def read_grid(fname, ngrid=None):
    """
    Read a 3D grid from a dat file.
    """

    with open(fname, "rb") as f:
        grid = np.fromfile(f, dtype="f", count=-1)

    if ngrid is not None:
        return grid.reshape(ngrid, ngrid, ngrid)
    else:
        return grid


def slice_2d(datacube, ngrid, nproj=None, operation="sum", axis=2):
    """
    Produces a slice from a 3D data cube for power spectra calculation.

    nproj: number of cells to project.

    operation suggestion either to "sum" or "mean" over projection cells.

    axis: The axis along which projection is done. It can be 1, 2 or 3.
    """

    ndim = np.ndim(datacube)
    # print("The dimension of data", ndim)

    if nproj == None:
        data_cut = datacube  # Project all cells along the axis

    else:
        if ndim == 1:
            data_cut = datacube.reshape(ngrid, ngrid, ngrid)[
                :, :, :nproj
            ]  # Project first nproj cells along the axis

        elif ndim == 3:
            data_cut = datacube[
                :, :, :nproj
            ]  # Project first nproj cells along the axis

        else:
            raise ValueError(
                "Provide the data either in 1D or 3D data cube (in case of projection)"
            )

    if operation == "sum":
        # Project number of cells along the third axis.
        data_2d = data_cut.sum(axis=axis)
    if operation == "mean":
        data_2d = data_cut.mean(axis=axis)

    return data_2d


def make_slice_2d(datacube, ngrid, nproj, operation="sum", axis=2):
    """
    Produces a slice from a 3D data cube for power spectra calculation.

    nproj: number of cells to project.

    operation suggestion either to "sum" or "mean" over projection cells.

    axis: The axis along which projection is done. It can be 1, 2 or 3.
    """

    ndim = np.ndim(datacube)
    # print("The dimension of data", ndim)

    if ndim == 1:
        data_cut = datacube.reshape(ngrid, ngrid, ngrid)[:, :, :nproj]

    elif ndim == 3:
        data_cut = datacube[:, :, :nproj]

    else:
        raise ValueError(
            "Provide the data either in 1D or 3D data cube (in case of projection)"
        )

    if operation == "sum":
        # Project number of cells along the third axis.
        data_2d = data_cut.sum(axis=axis)
    if operation == "mean":
        data_2d = data_cut.mean(axis=axis)

    return data_2d


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
    data = np.loadtxt(hlist_path_ascii)

    # save the file in npz format either in mentioned filename or in original ascii filename
    if filename:
        np.savez(filename, m=data[:, 0], x=data[:, 1], y=data[:, 2], z=data[:, 3])
    else:
        np.savez(
            hlist_path_ascii, m=data[:, 0], x=data[:, 1], y=data[:, 2], z=data[:, 3]
        )
    return


def make_halocat(halo_file, halocat_type="input_cat", mmin=None, boxsize=None):
    """
    reads the mass and co-ordinates of halos from a npz file.

    boxsize in Mpc unit according to 21cmfast convention.

    Input: halo file in npz format

    Returns: m and (x,y,z)
    """

    if halocat_type == "input_cat":
        fn = np.load(halo_file)
        if mmin == None:
            # halomass and x,y,z are read in the following format
            halomass, halo_x, halo_y, halo_z = fn["m"], fn["x"], fn["y"], fn["z"]

        else:
            halomass, halo_x, halo_y, halo_z = fn["m"], fn["x"], fn["y"], fn["z"]
            mass_cut = halomass >= mmin
            halomass = halomass[mass_cut]
            halo_x = halo_x[mass_cut]
            halo_y = halo_y[mass_cut]
            halo_z = halo_z[mass_cut]

            ind_sort = np.argsort(halomass)
            halomass = halomass[ind_sort]
            halo_x = halo_x[ind_sort]
            halo_y = halo_y[ind_sort]
            halo_z = halo_z[ind_sort]

    halo_cm = np.array([halo_x, halo_y, halo_z]).T

    del halo_x
    del halo_y
    del halo_z

    return halomass, halo_cm


"""
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
 """


def cic(x):
    dx, f = np.modf(x)
    return int(f), (1.0 - dx, dx)


def grid(posarr, weight, boxlength, ngrid, ndim=3):
    cell_size = boxlength / ngrid
    Vcell = cell_size**ndim

    n = ngrid
    gridarr = np.zeros(n**3)
    for i in range(weight.size):
        x = posarr[3 * i] / cell_size
        y = posarr[3 * i + 1] / cell_size
        z = posarr[3 * i + 2] / cell_size
        fx, cx = cic(x)
        fy, cy = cic(y)
        fz, cz = cic(z)
        for i1 in range(2):
            j1 = i1 + fx if i1 + fx < n else 0
            # print(j1)
            for i2 in range(2):
                j2 = i2 + fy if i2 + fy < n else 0
                for i3 in range(2):
                    j3 = i3 + fz if i3 + fz < n else 0
                    gridarr[j3 + n * (j2 + n * j1)] += (
                        cx[i1] * cy[i2] * cz[i3] * weight[i] / Vcell
                    )
    return gridarr


def grid_obj(posarr, weight, boxlength, ngrid, ndim=3):
    cell_size = boxlength / ngrid
    Vcell = cell_size**ndim

    n = ngrid
    gridarr = np.zeros(n**3)
    for i in range(weight.size):
        x = posarr[3 * i] / cell_size
        y = posarr[3 * i + 1] / cell_size
        z = posarr[3 * i + 2] / cell_size
        fx, cx = cic(x)
        fy, cy = cic(y)
        fz, cz = cic(z)
        for i1 in range(2):
            j1 = i1 + fx if i1 + fx < n else 0
            # print(j1)
            for i2 in range(2):
                j2 = i2 + fy if i2 + fy < n else 0
                for i3 in range(2):
                    j3 = i3 + fz if i3 + fz < n else 0
                    gridarr[j3 + n * (j2 + n * j1)] += (
                        cx[i1] * cy[i2] * cz[i3] * weight[i] / Vcell
                    )
    return gridarr


def grid_2d(posarr, weight, boxlength, ngrid, ndim=2):
    cell_size = boxlength / ngrid
    Vcell = cell_size**ndim

    n = ngrid
    gridarr = np.zeros(n**2)
    for i in range(weight.size):
        x = posarr[2 * i] / cell_size
        y = posarr[2 * i + 1] / cell_size
        fx, cx = cic(x)
        fy, cy = cic(y)
        for i1 in range(2):
            j1 = i1 + fx if i1 + fx < n else 0
            # print(j1)
            for i2 in range(2):
                j2 = i2 + fy if i2 + fy < n else 0
                gridarr[j2 + n * j1] += cx[i1] * cy[i2] * weight[i] / Vcell
    return gridarr


def make_grid(hloc, output_grid_dim="2D", weight=None, boxsize=None, ngrid=None):
    cellsize = boxsize / ngrid

    if output_grid_dim == "2D":
        grid = np.zeros([ngrid, ngrid])

        if np.isscalar(weight) == True:
            weight = weight * np.ones(len(hloc))

        for i in range(len(hloc)):
            hx = int(hloc[i][0] / cellsize)
            hy = int(hloc[i][1] / cellsize)

            grid[hx][hy] += weight[i]

            
    elif output_grid_dim == "3D":
        grid = np.zeros([ngrid, ngrid, ngrid])
        if np.isscalar(weight) == True:
            weight = weight * np.ones(len(hloc))

        for i in range(len(hloc)):
            hx = int(hloc[i][0] / cellsize)
            hy = int(hloc[i][1] / cellsize)
            hz = int(hloc[i][2] / cellsize)         
            grid[hx][hy][hz] += weight[i]

    return grid



def make_grid_new(hloc, output_grid_dim="2D", weight=None, boxsize=None, ngrid=None):
    cellsize = boxsize / ngrid

    if output_grid_dim == "2D":
        grid = np.zeros([ngrid, ngrid])

        if np.isscalar(weight) == True:
            weight = weight * np.ones(len(hloc))

        for i in range(len(hloc)):
            hx = int(hloc[i][0] / cellsize)
            hy = int(hloc[i][1] / cellsize)

            grid[hx][hy] += weight[i]

            

    elif output_grid_dim == "3D":
        grid = np.zeros([ngrid, ngrid, ngrid])
        if np.isscalar(weight) == True:
            weight = weight * np.ones(len(hloc))

        for i in range(len(hloc)):
            hx = int(hloc[i][0] / cellsize)
            hy = int(hloc[i][1] / cellsize)
            hz = int(hloc[i][2] / cellsize)

            grid[hx][hy][hz] += weight[i]


    return grid


def make_grid_rectangular(
    hloc,
    weight=None,
    ngrid_x=None,
    ngrid_y=None,
    ngrid_z=None,
    boxsize_x=None,
    boxsize_y=None,
    boxsize_z=None,
):

    if ngrid_x == None and ngrid_y == None and ngrid_z == None:
        print("Specify the grid numbers along x, y, z directions")

    if boxsize_x == None and boxsize_y == None and boxsize_z == None:
        print("Specify the box size in MPc along x, y, z directions")

    cellsize_x = boxsize_x / ngrid_x
    cellsize_y = boxsize_y / ngrid_y
    cellsize_z = boxsize_z / ngrid_z

    grid = np.zeros([ngrid_x, ngrid_y, ngrid_z])
    if np.isscalar(weight) == True:
        weight = weight * np.ones(len(hloc))

    for i in range(len(hloc)):
        hx = int(hloc[i][0] / cellsize_x)
        hy = int(hloc[i][1] / cellsize_y)
        hz = int(hloc[i][2] / cellsize_z)

        grid[hx][hy][hz] += weight[i]

    return grid


def make_grid_rectangular_object_number(
    hloc,
    weight=None,
    ngrid_x=None,
    ngrid_y=None,
    ngrid_z=None,
    boxsize_x=None,
    boxsize_y=None,
    boxsize_z=None,
):

    if ngrid_x == None and ngrid_y == None and ngrid_z == None:
        print("Specify the grid numbers along x, y, z directions")

    if boxsize_x == None and boxsize_y == None and boxsize_z == None:
        print("Specify the box size in MPc along x, y, z directions")

    cellsize_x = boxsize_x / ngrid_x
    cellsize_y = boxsize_y / ngrid_y
    cellsize_z = boxsize_z / ngrid_z

    grid = np.zeros([ngrid_x, ngrid_y, ngrid_z])

    if np.isscalar(weight) == True:
        weight = weight * np.ones(len(hloc))

    if weight is None:
        weight = np.ones(len(hloc))

    for i in range(len(hloc)):
        hx = int(hloc[i][0] / cellsize_x)
        hy = int(hloc[i][1] / cellsize_y)
        hz = int(hloc[i][2] / cellsize_z)
        grid[hx][hy][hz] += 1

    return grid


def update_params(params_old, params_to_update):
    for i in params_old:
        for j in params_to_update:
            if i == j:
                params_old[i] = params_to_update[j]
    return params_old


def make_grid_3D(hloc, weight=None, boxsize=None, ngrid=None):
    cellsize = boxsize / ngrid

    grid = np.zeros([ngrid, ngrid, ngrid])
    if np.isscalar(weight) == True:
        weight = weight * np.ones(len(hloc))

    for i in range(len(hloc)):
        hx = int(hloc[i][0] / cellsize)
        hy = int(hloc[i][1] / cellsize)
        hz = int(hloc[i][2] / cellsize)
        
        grid[hx][hy][hz] += weight[i]

    return grid



def get_noise_grid(pk_noise, 
                  boxsize_x=None,
                  boxsize_y=None,
                  boxsize_z=None,
                  ngrid_x=None,
                  ngrid_y=None, 
                  ngrid_z=None):

    # This function calculates a noise grid based using Gaussian approximation for a simulation box.

    x_ngrid= ngrid_x
    y_ngrid= ngrid_y
    z_ngrid= ngrid_z

    numbers_norm=np.random.normal(0.0, 1, x_ngrid * y_ngrid * z_ngrid)
    num_norm=numbers_norm.reshape( x_ngrid , y_ngrid , z_ngrid)

    Vol=(boxsize_x/ x_ngrid)  * (boxsize_y/ y_ngrid) * (boxsize_z/ z_ngrid)

    print("Calculating Fourier Transform to generate noise grid")
    f1=np.sqrt(pk_noise)*np.fft.fftn(num_norm)/np.sqrt(Vol)

    print("Calculating Inverse Fourier Transform to generate noise grid")
    f1_rp=np.fft.ifftn(f1)
    f2=np.real(f1_rp)
    
    return f2



def dk(k, pk):
    return k **3 * pk/2.0/np.pi**2




