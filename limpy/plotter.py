#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:07:29 2024

@author: anirbanroy
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import limpy.utils as lu
import limpy.lines as ll

class plot:
    @staticmethod
    def plot_beam_convolution(
        convolved_grid,
        ngrid,
        boxsize,
        halo_redshift,
        plot_unit="mpc",
        quantity="intensity",
        cmap="gist_heat",
        tick_num=5,
        vmin=None,
        vmax=None,
        title="",
        plot_scale="log",
    ):
        fig, ax = plt.subplots(figsize=(7, 7), dpi=100)

        radian_to_minute = (180.0 * 60.0) / np.pi
        radian_to_degree = (180.0) / np.pi

        if vmin is None:
            vmin = 0.1
        if vmax is None:
            vmax = convolved_grid.max()

        if plot_scale == "log":
            convolved_grid[
                convolved_grid <= 0
            ] = 1e-20  # Fill zero or negative values with a very small number so that log(0) does not exist.
            res = ax.imshow(
                np.log10(convolved_grid),
                cmap=cmap,
                interpolation="gaussian",
                origin="lower",
                rasterized=True,
                alpha=0.9,
                vmin=np.log10(vmin),
                vmax=np.log10(vmax),
            )
            plt.title(title)
            if quantity == "intensity":
                colorbar_label = r"$\mathrm{log}\,I_{\rm line}$"
            if quantity == "luminosity":
                colorbar_label = r"$\mathrm{log}\,L_{\rm line}$"

        if plot_scale == "lin":
            res = ax.imshow(
                convolved_grid,
                cmap=cmap,
                interpolation="gaussian",
                origin="lower",
                rasterized=True,
                alpha=0.9,
                vmin=vmin,
                vmax=vmax,
            )
            plt.title(title)
            colorbar_label = r"$L_{\rm line}$"

            if quantity == "intensity":
                colorbar_label = r"$I_{\rm line}$"
            if quantity == "luminosity":
                colorbar_label = r"$L_{\rm line}$"

        if plot_unit == "degree":
            x_tick = (
                lu.comoving_boxsize_to_angle(halo_redshift, boxsize)
            ) * radian_to_degree
            cell_size = x_tick / ngrid
            ticks = np.linspace(0, x_tick, num=tick_num)
            labels = [str("{:.1f}".format(xx)) for xx in ticks]
            locs = [xx / cell_size for xx in ticks]
            plt.xlabel(r"$\Theta\,(\mathrm{degree})$")
            plt.ylabel(r"$\Theta\,(\mathrm{degree})$")

        if plot_unit == "minute":
            x_tick = (
                lu.comoving_boxsize_to_angle(halo_redshift, boxsize)
            ) * radian_to_minute
            cell_size = x_tick / ngrid
            ticks = np.linspace(0, x_tick, num=tick_num)
            labels = [str("{:.1f}".format(xx)) for xx in ticks]
            locs = [xx / cell_size for xx in ticks]
            plt.xlabel(r"$\Theta\,(\mathrm{arc-min})$")
            plt.ylabel(r"$\Theta\,(\mathrm{arc-min})$")

        if plot_unit == "mpc":
            x_tick = boxsize
            cell_size = boxsize / ngrid
            ticks = np.linspace(0, x_tick, num=tick_num)
            labels = [str("{:.1f}".format(xx)) for xx in ticks]
            locs = [xx / cell_size for xx in ticks]
            plt.xlabel(r"$X\,(\mathrm{Mpc})$")
            plt.ylabel(r"$Y\,(\mathrm{Mpc})$")

        plt.xticks(locs, labels)
        plt.yticks(locs, labels)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(res, cax=cax)
        cb.set_label(colorbar_label, labelpad=20)
        cb.solids.set_edgecolor("face")
        cb.ax.tick_params("both", which="major", length=3, width=1, direction="out")

        plt.tight_layout()
        plt.savefig("luminosity_beam.png")
        
        
        
    @staticmethod
    def plot_slice_zoomin(
        convolved_grid,
        ngrid,
        boxsize,
        halo_redshift,
        origin=[0, 0],
        cmap="hot",
        slice_size=5,
        plot_unit="mpc",
        quantity="intensity",
        tick_num=5,
        vmin=None,
        vmax=None,
        plot_scale="log",
    ):
        fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
        radian_to_minute = (180.0 * 60.0) / np.pi
        radian_to_degree = (180.0) / np.pi
        cellsize_mpc = boxsize / ngrid

        if vmin == None:
            vmin = 0.1
        if vmax == None:
            vmax = convolved_grid.max()

        slice_szie_to_ngrid = int(slice_size / cellsize_mpc)

        if origin == [0, 0]:
            x_grid = slice_szie_to_ngrid
            y_grid = x_grid
            convolved_grid = convolved_grid[0 : x_grid + 1, 0 : y_grid + 1]

        else:
            x_grid_origin = int(origin[0] / cellsize_mpc)
            y_grid_origin = int(origin[1] / cellsize_mpc)

            x_grid = x_grid_origin + slice_szie_to_ngrid
            y_grid = y_grid_origin + slice_szie_to_ngrid
            convolved_grid = convolved_grid[
                x_grid_origin : x_grid + 1, y_grid_origin : y_grid + 1
            ]

        if plot_scale == "log":
            convolved_grid[
                convolved_grid <= 0
            ] = 1e-20  # Fill zero or negative values with a very small number so that log(0) does not exist.
            res = ax.imshow(
                np.log10(convolved_grid),
                cmap=cmap,
                interpolation="gaussian",
                origin="lower",
                rasterized=True,
                alpha=0.9,
                vmin=np.log10(vmin),
                vmax=np.log10(vmax),
            )
            if quantity == "intensity":
                colorbar_label = r"$\mathrm{log}\,I_{\rm line}$"
            if quantity == "luminosity":
                colorbar_label = r"$\mathrm{log}\,L_{\rm line}$"

        if plot_scale == "lin":
            res = ax.imshow(
                convolved_grid,
                cmap=cmap,
                interpolation="gaussian",
                origin="lower",
                rasterized=True,
                alpha=0.9,
                vmin=vmin,
                vmax=vmax,
            )
            colorbar_label = r"$L_{\rm line}$"

            if quantity == "intensity":
                colorbar_label = r"$I_{\rm line}$"
            if quantity == "luminosity":
                colorbar_label = r"$L_{\rm line}$"

        if plot_unit == "degree":
            x_tick = (
                lu.comoving_boxsize_to_angle(halo_redshift, boxsize)
            ) * radian_to_degree
            cell_size = x_tick / ngrid
            ticks = np.linspace(0, x_tick, num=tick_num)
            labels = [str("{:.1f}".format(xx)) for xx in ticks]
            locs = [xx / cell_size for xx in ticks]
            plt.xlabel(r"$\Theta\,(\mathrm{degree})$")
            plt.ylabel(r"$\Theta\,(\mathrm{degree})$")

        if plot_unit == "minute":
            x_tick = (
                lu.comoving_boxsize_to_angle(halo_redshift, boxsize)
            ) * radian_to_minute
            cell_size = x_tick / ngrid
            ticks = np.linspace(0, x_tick, num=tick_num)
            labels = [str("{:.1f}".format(xx)) for xx in ticks]
            locs = [xx / cell_size for xx in ticks]
            plt.xlabel(r"$\Theta\,(\mathrm{arc-min})$")
            plt.ylabel(r"$\Theta\,(\mathrm{arc-min})$")

        if plot_unit == "mpc":
            x_tick = slice_size
            cell_size = boxsize / ngrid
            ticks = np.linspace(0, x_tick, num=tick_num)
            labels = [str("{:.1f}".format(xx)) for xx in ticks]
            locs = [xx / cell_size for xx in ticks]
            plt.xlabel(r"$X\,(\mathrm{Mpc})$")
            plt.ylabel(r"$Y\,(\mathrm{Mpc})$")

        plt.xticks(locs, labels)
        plt.yticks(locs, labels)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(res, cax=cax)
        cb.set_label(colorbar_label, labelpad=20)
        cb.solids.set_edgecolor("face")
        cb.ax.tick_params("both", which="major", length=3, width=1, direction="out")

        plt.tight_layout()
        plt.show()
        
        

    @staticmethod
    def plot_slice(
        boxsize,
        ngrid,
        nproj,
        dens_gas_file,
        halocat_file,
        halo_redshift,
        line_name="CII158",
        halocat_type="input_cat",
        halo_cutoff_mass=1e11,
        use_scatter=True,
        density_plot=False,
        halo_overplot=False,
        plot_lines=False,
        unit="mpc",
        line_cb_min=1e2,
        line_cb_max=1e10,
        params_fisher=None,
    ):
        low_mass_log = 0.0
        cellsize = boxsize / ngrid

        fig = plt.figure(figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        if density_plot:
            # Load density file
            with open(dens_gas_file, "rb") as f:
                dens_gas = np.fromfile(f, dtype="f", count=-1)
                # dens_gas=dens_gas+1.0
                # dens_gas=dens_gas.reshape(ngrid,ngrid,ngrid)
                # rhobar = np.mean(dens_gas)

            # slice the data cube
            i, j, val = slice(dens_gas, ngrid, nproj)

            dens_mean = np.mean(dens_gas + 1)

            val = val / (dens_mean * nproj)

            cellsize = boxsize / ngrid
            i *= cellsize
            j *= cellsize

            s = plt.scatter(
                i,
                j,
                c=val,
                s=10,
                marker="s",
                edgecolor="none",
                rasterized=True,
                cmap="magma",
                vmax=0.5,
                vmin=-0.5,
            )

            if unit == "mpc":
                ax.set_xlim(0, boxsize)
                ax.set_ylim(0, boxsize)
                plt.xlabel("cMpc/h")
                plt.ylabel("cMpc/h")

            elif unit == "degree":
                ax.set_xlim(0, boxsize)
                ax.set_ylim(0, boxsize)

                xmin = 0
                ymin = 0
                xmax = ymax = lu.comoving_boxsize_to_degree(halo_redshift, boxsize)

                N = 4
                xtick_mpc = ytick_mpc = np.linspace(0, boxsize, N)

                custom_yticks = np.round(np.linspace(ymin, ymax, N, dtype=float), 1)

                ax.set_yticks(ytick_mpc)
                ax.set_yticklabels(custom_yticks)

                custom_xticks = np.round(np.linspace(xmin, xmax, N, dtype=float), 1)
                ax.set_xticks(xtick_mpc)
                ax.set_xticklabels(custom_xticks)

                plt.xlabel(r"$\Theta\,(\mathrm{degree})$")
                plt.ylabel(r"$\Theta\,(\mathrm{degree})$")

            divider = make_axes_locatable(ax)

            cax = divider.append_axes("bottom", "3%", pad="13%")
            cb = plt.colorbar(s, cax=cax, orientation="horizontal")
            cb.set_label(r"$\Delta_\rho$", labelpad=5)

            cb.solids.set_edgecolor("face")
            ax.set_aspect("equal", "box")

        if halo_overplot:
            # Load density file
            with open(dens_gas_file, "rb") as f:
                dens_gas = np.fromfile(f, dtype="f", count=-1)
                # dens_gas=dens_gas+1.0
                # dens_gas=dens_gas.reshape(ngrid,ngrid,ngrid)
                # rhobar = np.mean(dens_gas)

            # slice the data cube
            i, j, val = slice(dens_gas, ngrid, nproj)

            dens_mean = np.mean(dens_gas + 1)

            val = val / (dens_mean * nproj)

            cellsize = boxsize / ngrid
            i *= cellsize
            j *= cellsize

            s = plt.scatter(
                i,
                j,
                c=val,
                s=10,
                marker="s",
                edgecolor="none",
                rasterized=True,
                cmap=plt.cm.viridis_r,
                vmax=1,
                vmin=-1,
            )

            halomass, halo_cm = lu.make_halocat(
                halocat_file,
                mmin=halo_cutoff_mass,
                halocat_type=halocat_type,
                boxsize=boxsize,
            )

            nhalo = len(halomass)
            # Overplot halos
            x_halos = halo_cm[range(0, nhalo * 3, 3)]
            y_halos = halo_cm[range(1, nhalo * 3, 3)]
            z_halos = halo_cm[range(2, nhalo * 3, 3)]

            print("Minimum halo mass:", halomass.min())
            print("Maximum halo mass:", halomass.max())

            # halomass_filter=halomass
            # logmh=np.log10(halomass)
            # logmh=np.array([int(logmh[key]) for key in range(nhalo)])

            # highmass_filter=np.where(logmh>halo_cutoff_mass,logmh,low_mass_log)

            highmass_filter = np.where(halomass > halo_cutoff_mass, halomass, low_mass_log)

            # z_min = 0.0
            z_max = nproj * cellsize  # See slice() above

            mask = z_halos < z_max
            x_halos = x_halos[mask]
            y_halos = y_halos[mask]
            r = highmass_filter[mask]
            r = r / r.max()

            s1 = plt.scatter(
                x_halos, y_halos, marker="o", s=100 * r, color="red", alpha=0.9
            )

            if unit == "mpc":
                ax.set_xlim(0, boxsize)
                ax.set_ylim(0, boxsize)
                plt.xlabel("cMpc/h")
                plt.ylabel("cMpc/h")

            elif unit == "degree":
                ax.set_xlim(0, boxsize)
                ax.set_ylim(0, boxsize)

                xmin = 0
                ymin = 0
                xmax = ymax = lu.comoving_boxsize_to_degree(halo_redshift, boxsize)

                N = 4
                xtick_mpc = ytick_mpc = np.linspace(0, boxsize, N)

                custom_yticks = np.round(np.linspace(ymin, ymax, N, dtype=float), 1)

                ax.set_yticks(ytick_mpc)
                ax.set_yticklabels(custom_yticks)

                custom_xticks = np.round(np.linspace(xmin, xmax, N, dtype=float), 1)
                ax.set_xticks(xtick_mpc)
                ax.set_xticklabels(custom_xticks)

                plt.xlabel(r"$\Theta\,(\mathrm{degree})$")
                plt.ylabel(r"$\Theta\,(\mathrm{degree})$")

            divider = make_axes_locatable(ax)

            cax = divider.append_axes("bottom", "3%", pad="13%")
            cb = plt.colorbar(s, cax=cax, orientation="horizontal")
            cb.set_label(r"$\Delta_\rho$", labelpad=5)

            cb.solids.set_edgecolor("face")
            ax.set_aspect("equal", "box")

        if plot_lines:
            # Load density file
            with open(dens_gas_file, "rb") as f:
                dens_gas = np.fromfile(f, dtype="f", count=-1)
                # dens_gas=dens_gas+1.0
                # dens_gas=dens_gas.reshape(ngrid**3,)
                # rhobar = np.mean(dens_gas)

            # Plot gas density
            i, j, val = slice(dens_gas, ngrid, nproj)
            # val = val/(rhobar*nproj)

            cellsize = boxsize / ngrid
            i *= cellsize
            j *= cellsize

            dens_mean = np.mean(dens_gas + 1)

            val = val / (dens_mean * nproj)

            s = plt.scatter(
                i,
                j,
                c=val,
                s=10,
                marker="s",
                edgecolor="none",
                rasterized=False,
                cmap="viridis_r",
                vmax=1,
                vmin=-1,
                alpha=0.9,
            )

            xl, yl, lum = ll.calc_luminosity(
                boxsize,
                ngrid,
                nproj,
                halocat_file,
                halo_redshift,
                line_name=line_name,
                halo_cutoff_mass=halo_cutoff_mass,
                halocat_type=halocat_type,
                use_scatter=use_scatter,
                unit="mpc",
            )

            r = (np.log10(lum) / np.log10(lum.max())) ** 6

            s1 = plt.scatter(
                xl,
                yl,
                marker="o",
                c=lum,
                s=70 * r,
                cmap="afmhot",
                vmin=line_cb_min,
                vmax=line_cb_max,
                norm=plt.colors.LogNorm(),
                alpha=0.9,
            )

            if unit == "mpc":
                ax.set_xlim(0, boxsize)
                ax.set_ylim(0, boxsize)
                plt.xlabel("cMpc")
                plt.ylabel("cMpc")

            elif unit == "degree":
                ax.set_xlim(0, boxsize)
                ax.set_ylim(0, boxsize)

                xmin = 0
                ymin = 0
                xmax = ymax = lu.comoving_boxsize_to_degree(halo_redshift, boxsize)

                N = 4
                xtick_mpc = ytick_mpc = np.linspace(0, boxsize, N)

                custom_yticks = np.round(np.linspace(ymin, ymax, N, dtype=float), 1)

                ax.set_yticks(ytick_mpc)
                ax.set_yticklabels(custom_yticks)

                custom_xticks = np.round(np.linspace(xmin, xmax, N, dtype=float), 1)
                ax.set_xticks(xtick_mpc)
                ax.set_xticklabels(custom_xticks)

                plt.xlabel(r"$\Theta\,(\mathrm{degree})$")
                plt.ylabel(r"$\Theta\,(\mathrm{degree})$")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", "5%", pad="3%")
            cb = plt.colorbar(s1, cax=cax)
            cb.set_label(r"$L_{\mathrm{%s}}\,[L_\odot]$" % (line_name), labelpad=1)

            cb.solids.set_edgecolor("face")
            ax.set_aspect("equal", "box")

            cax1 = divider.append_axes("bottom", "3%", pad="13%")
            cb1 = plt.colorbar(s, cax=cax1, orientation="horizontal")
            cb1.set_label(r"$\Delta_\rho$", labelpad=5)
            cb1.solids.set_edgecolor("face")

        plt.tight_layout()
        plt.savefig("slice_plot.pdf", bbox_inches="tight")


        
        
        
        

        
        
        
    