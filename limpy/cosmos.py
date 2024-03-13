import camb
import numpy as np
import scipy.integrate as si
from camb import get_matter_power_interpolator
from colossus.cosmology import cosmology as col_cosmology
from colossus.lss import bias, mass_function


import limpy.inputs as inp

class cosmo:
    def __init__(self, parameters=None):
        
        if parameters is None:
            parameters = inp.parameters_default 
        else:
            parameters = {**inp.parameters_default, **parameters}
            
        # Initialize cosmological parameters with inputs
        self.h = parameters['h']
        self.omega_lambda = parameters['omega_lambda']
        self.omega_b = parameters['omega_b']
        self.omega_m = parameters['omega_m']
        self.omega_cdm = self.omega_m - self.omega_b
        self.ns = parameters['ns']
        self.sigma_8 = parameters['sigma_8']

        self.H_0 = 100 * self.h  # Km/S/Mpc

        # Handle omega_k separately
        self.omega_k = 1 - (self.omega_m + self.omega_lambda)

        # Handle astrophysical parameters using variable names
        self.M_min = parameters['M_min']
        self.M_max = parameters['M_max']
        self.delta_c = parameters['delta_c']
        self.halo_model = parameters['halo_model']
        self.halo_mass_def = parameters['halo_mass_def']
        self.bias_model = parameters['bias_model']
        self.bias_mass_def = parameters['bias_mass_def']
        
        #Initialize cosmological parameters for colossus
        #set_cosmo = self.set_cosmo_colossus()
        
        print("<---Parameters used in cosmo.py--->:")
        print("Hubble constant (h):", self.h)
        print("Omega matter (Omega_m):", self.omega_m)

    def E_z(self, z):
        return np.sqrt(self.omega_m * (1 + z) ** 3 + self.omega_k * (1 + z) ** 2 + self.omega_lambda)

    def H_z(self, z):
        """
        Hubble constant at redshift z.
        unit: s^{-1}
        """
        return (
            100
            * self.h
            * np.sqrt(self.omega_m * (1 + z) ** 3 + self.omega_k * (1 + z) ** 2 + self.omega_lambda)
            * inp.km_to_m
            / inp.mpc_to_m
        )

    def D_co_unvec(self, z):
        """
        Comoving distance transverse.
        """
        omega_k_abs = abs(self.omega_k)
        D_H = inp.c_in_mpc / (self.H_0 * inp.km_to_mpc) 
        
        D_H *= self.h # in Mpc/h unit

        D_c_int = lambda z: D_H / self.E_z(z)
        D_c = si.quad(D_c_int, 0, z, limit=1000)[0] 

        if self.omega_k == 0:
            return D_c
        elif self.omega_k < 0:
            return D_H / np.sqrt(omega_k_abs) * np.sin(np.sqrt(omega_k_abs) * D_c / D_H)
        elif self.omega_k > 0:
            return D_H * np.sinh(np.sqrt(self.omega_k) * D_c / D_H) / np.sqrt(self.omega_k) 
        
        
    def D_co(self, z):
        if np.isscalar(z):
            return self.D_co_unvec(z)  # Mpc unit
        else:
            result_array = np.zeros(len(z))
            for i in range(len(z)):
                result_array[i] = self.D_co_unvec(z[i])

            return result_array  # Mpc/h unit

    def z_co_unvec(self, z):
        """
        Comoving distance transverse.
        """
        omega_k_abs = abs(self.omega_k)
        D_H = inp.c_in_mpc / (self.H_0 * inp.km_to_mpc)
        #D_H *= self.h 

        D_c_int = lambda z: D_H / self.E_z(z)
        D_c = si.quad(D_c_int, 0, z, limit=1000)[0]

        if self.omega_k == 0:
            return D_c
        elif self.omega_k < 0:
            return D_H / np.sqrt(omega_k_abs) * np.sin(np.sqrt(omega_k_abs) * D_c / D_H)
        elif self.omega_k > 0:
            return D_H * np.sinh(np.sqrt(self.omega_k) * D_c / D_H) / np.sqrt(self.omega_k)

    def D_angular(self, z):
        """
        Angular diameter distance
        """
        if np.isscalar(z):
            return self.D_co_unvec(z) / (1 + z)
        else:
            return [self.D_co_unvec(zin) / (1 + zin) for zin in z]

    
    def D_luminosity(self, z):
        """
        Angular diameter distance
        """
        if np.isscalar(z):
            return self.D_angular(z) * (1 + z) ** 2
        else:
            result_array = np.zeros(len(z))
            for i in range(len(z)):
                result_array[i] = self.D_angular(z[i]) * (1 + z[i]) ** 2

            return result_array
        
        
    def solid_angle(self, length, z):
        "Solid angle in Sr unit"
        A = length * length
        chi = self.D_co(z)
        return A / chi**2

    
    def pk_camb(self, k, z, kmax=10.0):
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=100 * self.h, ombh2=self.omega_b * self.h**2, omch2=self.omega_cdm * self.h**2,
            omk= self.omega_k
            
        )
        pars.InitPower.set_params(ns=self.ns)

        pars.set_matter_power(redshifts=[z], kmax=kmax)
        

        PK = get_matter_power_interpolator(pars)

        return PK.P(z, k)


    def set_cosmo_colossus(self):
        params = {
            "flat": False,
            "H0": 100 * self.h,
            "Ode0": self.omega_lambda,
            "Om0": self.omega_m,
            "Ob0": self.omega_b,
            "sigma8": self.sigma_8,
            "ns": self.ns,
        }
        #col_cosmology.addCosmology("myCosmo", params)
        #elf.cosmo_col = col_cosmology.setCosmology("myCosmo")
        return params

    
    def hmf_setup(
        self, z, q_out="dndlnM",
        halo_model=None,
        mdef=None,
    ):

        params= self.set_cosmo_colossus()
        
        col_cosmology.addCosmology("myCosmo", params)
        self.cosmo_col = col_cosmology.setCosmology("myCosmo")
        
        
        if halo_model is None:
            halo_model = self.halo_model

        if mdef is None:
            mdef = self.halo_mass_def
            
        Mh = 10 ** (
            np.linspace(np.log10(self.M_min), np.log10(self.M_max), num=2000))
        #  # M_sun/h unit
        
        #Mh = np.logspace(np.log10(self.M_min), np.log10(self.M_max), num=2000)
       

        mfunc = mass_function.massFunction(
            Mh, z, mdef=mdef, model=halo_model, q_out=q_out
        )

        return Mh, mfunc
    
    
    def bias_dm(self, m, z, bias_model=None, mdef=None):
        """
        Calculate the halo bias for dark matter.
        
        Parameters:
            m (float or array): Halo mass in units of M_sun/h.
            z (float or array): Redshift.
            bias_model (str, optional): Halo bias model (default: None).
            mdef (str, optional): Halo mass definition (default: None).
            
        Returns:
            float or array: Halo bias.
        """
        
        if bias_model is None:
            bias_model = self.bias_model
        if mdef is None:
            mdef = self.bias_mass_def

        b = bias.haloBias(m, z=z, model=bias_model, mdef=mdef)
        return b
    
    
    
    def angle_to_comoving_size(self, z, angle):
        """
        Get the comoving size for an angle at redshift z.

        Parameters
        ----------
        z: float
            Redshift
        angle: float
            Angle in radians.

        Returns
        -------
        size: float
            Size in Mpc/h.
        """
        dc = self.D_co(z)

        
        size = angle * dc
        return size
    
    
    
    def comoving_boxsize_to_angle(self, z, boxsize):
        """
        Angle subtended by the surface of a box at redshift z.

        Parameters
        ----------
        z: float
            Redshift
        boxsize: float
            The length of the box in Mpc/h.

        Returns
        -------
        theta_rad: float
            Angle in radians.
        """
        da = self.D_co(z)
        theta_rad = boxsize / da
        return theta_rad
    
    
    def angle_to_comoving_boxsize(self, z, angle, angle_unit="degree"):
        """
        Angle subtended by the surface of a box at redshift z.

        Parameters
        ----------
        z: float
            Redshift
        angle: float
            The angle in radians or degrees defined by angle_unit.
        angle_unit: str, optional
            The unit of angle, either "degree" or "radian".

        Returns
        -------
        boxsize: float
            The comoving box size in Mpc/h.
        """
        if angle_unit == "degree":
            theta_rad = angle * np.pi / 180
        elif angle_unit == "radian":
            theta_rad = angle
        else:
            raise ValueError("Invalid angle_unit. Use 'degree' or 'radian'.")

        da = self.D_co(z)
        boxsize = theta_rad * da
        return boxsize
    
    
    def physical_boxsize_to_angle(self, z, boxsize):
        """
        Angle subtended by the surface of a box at redshift z.

        Parameters
        ----------
        z: float
            Redshift
        boxsize: float
            The physical box size in Mpc/h.

        Returns
        -------
        angle: float
            The angle in radians subtended by the box at the given redshift.
        """
        da = self.D_angular(z)
        angle = boxsize / da
        return angle
    
    

    def length_projection(self, z=None, dz=None, nu_obs=None, dnu=None, line_name="CII"):
        """
        This function returns the projection length for the frequency resolution, dnu_obs.

        Parameters
        ----------
        z: float
            Redshift

        dz: float
            Redshift bin size. z and dz have to be passed together.

        nu_obs: float
            Observational frequency in GHz.

        dnu_obs: float
            Observational frequency resolution in GHz.

        line_name: str
            The name of the lines. Check "line_list" to get all the available lines.

        Returns
        -------
        boxsize: float
            The comoving box size in (Mpc/h)
        """

        if (z is None and dz is not None) or (z is not None and dz is None):
            raise ValueError(
                "Specify z and dz together to calculate the projection length calculation."
            )

        if (nu_obs is None and dnu is not None) or (nu_obs is not None and dnu is None):
            raise ValueError(
                "Specify nu_obs and dnu together to calculate the projection length calculation."
            )

        if (z is None and dz is None) and (nu_obs is None and dnu is None):
            raise ValueError(
                "Either specify z and dz together or nu_obs and dnu together for projection length calculation."
            )

        if z != None and dz != None:
            dco1 = self.D_co(z)
            dco2 = self.D_co(z + dz)
            res = dco2 - dco1

        if nu_obs != None and dnu != None:
            z_obs1 = inp.nu_obs_to_z(nu_obs, line_name=line_name)
            z_obs2 = inp.nu_obs_to_z((nu_obs + dnu), line_name=line_name)

            dco1 = self.D_co(z_obs1)
            dco2 = self.D_co(z_obs2)
            res = dco1 - dco2

        return res
    
    
    def box_freq_to_quantities(self, nu_obs=280, dnu_obs=2.8, boxsize=80, ngrid=512, z_start=None, line_name="CII158"):
        nu_rest = inp.nu_rest(line_name=line_name)
        cell_size = boxsize / ngrid
        
        if z_start:
            z_em = z_start
        else:
            z_em = (nu_rest / nu_obs) - 1
        
        dz_em = nu_rest * dnu_obs / (nu_obs * (nu_obs + dnu_obs))
        d_chi = self.D_co(z_em + dz_em) - self.D_co(z_em)
        d_ngrid = int(d_chi / cell_size)
        
        return round(z_em, 2), round(dz_em, 2), d_chi, d_ngrid
        

    def comoving_size_to_delta_nu(self, length, z, line_name="CII158"):
        nu_rest_line = inp.nu_rest(line_name=line_name)

        dchi_dz = inp.c_in_mpc / self.H_z(z)

        dnu = (length) * nu_rest_line / (dchi_dz * (1 + z) ** 2)

        return dnu
    

    def V_pix(self, theta_beam, delta_nu, beam_unit="arcmin", line_name="CII158"):
        """
        theta_beam: beam size in arc-min
        delta_nu: the frequency resolution in GHz
        line_name: name of the line
        """

        theta_rad = self.convert_beam_unit_to_radian(theta_beam, beam_unit=beam_unit)

        nu = inp.nu_rest(line_name)

        lambda_line = self.freq_to_lambda(nu)

        y = lambda_line * (1 + self.z) ** 2 / self.H_z(self.z)
        res = self.D_co(self.z) ** 2 * y * (theta_rad) ** 2 * delta_nu
        return res  # (Mpc/h)^3
