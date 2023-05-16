import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
from scipy.integrate import quadrature
from astropy.cosmology import Planck18, z_at_value
from astropy.constants import c
import astropy.units as u
import utils

class grid:
    def __init__(self, nside, n_samples_of_chi, z_min_int=0.005, z_max_int = 3.):
        self.nside = nside
        self.n_samples_of_chi = n_samples_of_chi
        self.chi_min_int = Planck18.comoving_distance(z_min_int).value
        self.chi_max_int = Planck18.comoving_distance(z_max_int).value
        self.chi_array = np.linspace(chi_min_int, chi_max_int + 100, n_samples_of_chi)
        # Get the redshift corresponding to these values of chi in the Planck18 cosmology
        self.z_array = np.zeros((hp.nside2npix(nside), n_samples_of_chi))
        for i, chi in enumerate(self.chi_array):
            self.z_array[:, i] = z_at_value(Planck18.comoving_distance, chi * u.Mpc)

    def hashdict(self):
        return {'n_samples_of_chi': self.n_samples_of_chi,
                'nside': self.nside, 'chi_min_int': self.chi_min_int, 'chi_max_int': self.chi_max_int}

    def __eq__(self, other):
        return self.compatible(other)

    def compatible(self, other):
        return ((self.n_samples_of_chi == other.n_samples_of_chi) and
                (self.nside == other.nside)  and
                (self.chi_min_int == other.chi_min_int) and
                (self.chi_max_int == other.chi_max_int))

class Field:
    def __init__(self, grid, p_pert_array, p_fid_array):

        self.delta_p_maps = p_pert_array - p_fid_array
        self.kernel = interp1d(grid.chi_array, p_fid_array)
        self.grid = grid

        # Take the spherical harmonic transform of each chi slice.
        # Conveniently, we can take complex SHT so array sizes reduce by x2
        self.delta_p_lm_of_chi = np.zeros((hp.Alm.getsize(3 * grid.nside - 1), grid.n_samples_of_chi), dtype=complex)
        for i in range(grid.n_samples_of_chi):
            self.delta_p_lm_of_chi[:, i] = hp.map2alm(self.delta_p_maps[:, i])

class gal_delta(Field):
    def __init__(self, grid, sigma, z_mean, template_zmean_shifts=0., template_width_shifts=0., get_delta_p=True):
        self.sigma = sigma
        self.z_mean = z_mean

        # Convert template of z-shifts to chi-shifts
        chimean_shifts_array = Planck18.comoving_distance(z_mean + template_zmean_shifts).value \
                               - Planck18.comoving_distance(z_mean).value
        width_shifts_array = Planck18.comoving_distance(z_mean + template_width_shifts).value \
                               - Planck18.comoving_distance(z_mean).value

        # In each pixel, calculate the perturbed dndz as a Gaussian in chi
        dndz_perturbed = (1 / ((grid.chi_sigma_fid + width_shifts_array[..., np.newaxis]) * np.sqrt(2 * np.pi))) * np.exp(
            -(grid.chi_array - grid.chi_mean_fid - chimean_shifts_array[..., np.newaxis]) ** 2 / (
                        2 * (grid.chi_sigma_fid + width_shifts_array[..., np.newaxis]) ** 2))
        # Take the fiducial dndz to be the monopole of the perturbed dndz
        dndz_fid = np.mean(dndz_perturbed, axis=0)

        # Convert dndz to selection function
        phi_perturbed_array = (Planck18.H(grid.z_array[0, :]) / c).value * dndz_perturbed
        phi_fid_array = (Planck18.H(grid.z_array[0, :]) / c).value * dndz_fid

        # Normalize the selection function so that \int_{0}^{inf} dr phi(r) = 1
        phi_norm, error = quadrature(interp1d(grid.chi_array, phi_fid_array), grid.chi_min_int, grid.chi_max_int,
                                     tol=1e-20, miniter=3000)
        phi_perturbed_array *= phi_norm ** (-1)
        phi_fid_array *= phi_norm ** (-1)

        if get_delta_p:
            super().__init__(phi_perturbed_array, phi_fid_array)
        else:
            self.phi_perturbed_array = phi_perturbed_array
            self.phi_fid_array = phi_fid_array

class gal_shear(Field):
    def __init__(self, grid, sigma, z_mean, template_zmean_shifts=None, template_width_shifts=None):
        self.sigma = sigma
        self.z_mean = z_mean

        g_d = gal_delta(grid, sigma, z_mean, grid, template_zmean_shifts, template_width_shifts, get_delta_p=False)
        phi_fid_array = g_d.phi_fid_array
        phi_perturbed_array = g_d.phi_perturbed_array

        # If doing cosmic shear, calculate the galaxy lensing kernel
        # Dummy run to pre-compile with numba
        utils.lens_efficiency_kernel(grid.chi_array, grid.chi_max_int, np.zeros_like(grid.chi_array))

        # Now in earnest
        g_fid = 3 / 2. * Planck18.Om0 * Planck18.H0.value ** 2 / c.value ** 2 * grid.chi_array * (
                1 + grid.z_array[0, :]) * utils.lens_efficiency_kernel(grid.chi_array, grid.chi_max_int, phi_fid_array)
        g_pert = 3 / 2. * Planck18.Om0 * Planck18.H0.value ** 2 / c.value ** 2 * grid.chi_array * (
                1 + grid.z_array[0, :]) * utils.lens_efficiency_kernel(grid.chi_array, grid.chi_max_int, phi_perturbed_array)

        super().__init__(grid, g_pert, g_fid)