import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
from scipy.integrate import quadrature
from astropy.cosmology import Planck18, z_at_value
from astropy.constants import c
import astropy.units as u
from cardiac import utils

class grid:
    def __init__(self, nside, n_samples_of_chi, z_min_int=0.005, z_max_int = 3.):
        """ Defines the numerical hyperparameters of our calculation
            - Inputs:sigma_chishift
                * nside = int, power of 2. Healpix nside defining the pixelization. Must match shifts template
                * z_min_int (optional) = float. Minimum of range for the integrals over chi
                * z_max_int (optional) = float. z_max_int
                * n_samples_of_chi = int. A power of 2, # of samples in comoving distance
        """
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.n_samples_of_chi = n_samples_of_chi
        self.chi_min_int = Planck18.comoving_distance(z_min_int).value
        self.chi_max_int = Planck18.comoving_distance(z_max_int).value
        self.chi_array = np.linspace(self.chi_min_int, self.chi_max_int + 100, n_samples_of_chi)
        # Get the redshift corresponding to these values of chi in the Planck18 cosmology
        self.z_array = np.zeros((self.npix, n_samples_of_chi))
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
    def __init__(self, grid, p_pert_array, p_fid_array, mask):
        """ General class defining observed fields subject to projection anisotropy. Computes analytic projection kernel
            and alms of each redshift slice
            - Inputs:
                * grid = Instance of grid class containing numerical hyperparams
                * p_pert_array = np.array of size (grid.npix, grid.n_samples_of_chi). Contains perturbation to phi/g/etc
                * p_fid_array = np.array of size (grid.n_samples_of_chi). Fiducial phi/g/etc
                * mask = np.array of float values between 0 and 1 defining observed footprint
        """
        self.delta_p_maps = mask[..., None] * (p_pert_array - p_fid_array)
        self.p_fid_array = p_fid_array
        self.grid = grid
        self.mask = mask

        # Take the spherical harmonic transform of each chi slice.
        # Conveniently, we can take complex SHT so array sizes reduce by x2
        self.delta_p_lm_of_chi = np.zeros((hp.Alm.getsize(3 * grid.nside - 1), grid.n_samples_of_chi), dtype=complex)
        for i in range(grid.n_samples_of_chi):
            self.delta_p_lm_of_chi[:, i] = hp.map2alm(self.delta_p_maps[:, i])

    # ToDo: Maybe 'equality' isn't the clearest way of checking this
    def __eq__(self, other):
        return self.compatible(other)

    def compatible(self, other):
        return self.grid == other.grid

class GalDelta(Field):
    def __init__(self, grid, sigma, z_mean, bvec, template_zmean_shifts=None, template_width_shifts=None,
                 template_interloper_frac=None, interloper_sigma=None, interloper_z_mean=None, get_delta_p=True):
        """ Observed galaxy clustering field subject to anisotropy in its (Gaussian) dN/dz
            - Inputs:
                * grid = Instance of grid class containing numerical hyperparams
                * sigma = float. Standard deviation of the fiducial dN/dz
                * z_mean = float. Central redshift of the fiducial dN/dz
                * bvec = list containing [b1,    b2,    bs2,   bnabla2, SN] to be fed to Anzu to obtain Pgg
                * template_zmean_shifts = instance of templates.Template with shifts in the mean redshift of the dN/dz
                * template_width_shifts = instance of templates.Template with shifts in the width of the dN/dz
                * template_interloper_frac = instance of templates.Template with fraction of main Gaussian that's in
                                             the secondary one
                * interloper_sigma = float. Standard deviation of the fiducial interloper dN/dz
                * interloper_z_mean = float. Central redshift of the fiducial interloper dN/dz
                * get_delta_p (optional) = Bool. If False, use as helper function in cosmic shear calculation
        """
        if template_zmean_shifts is not None:
            assert (grid.npix==len(template_zmean_shifts.map)), "grid does not match nside of z_mean shift template"
            self.mask = template_zmean_shifts.mask
        if template_width_shifts is not None:
            assert (grid.npix==len(template_width_shifts.map)), "grid does not match nside of width shift template"
            self.mask = template_width_shifts.mask
        if template_interloper_frac is not None:
            assert (interloper_sigma is not None and interloper_z_mean is not None), "Provide interloper dN/dz!"
            assert (grid.npix==len(template_interloper_frac.map)), "grid does not match nside of interloped f. template"
            self.mask = template_interloper_frac.mask

        self.template_zmean_shifts = template_zmean_shifts
        self.template_width_shifts = template_width_shifts
        self.template_interloper_frac = template_interloper_frac
        self.sigma = sigma
        self.z_mean = z_mean
        self.interloper_sigma = interloper_sigma
        self.interloper_z_mean = interloper_z_mean
        self.bvec = bvec # ToDo: Allow tracers to have different bias
        # The user input is in redshift units because this is more intuitive. However, we will define our dndzs to be
        # Gaussian in comoving distance. So next, we convert to chi
        self.chi_mean_fid = Planck18.comoving_distance(z_mean).value
        # Width of the fiducial distribution
        self.chi_sigma_fid = Planck18.comoving_distance(z_mean + sigma).value - Planck18.comoving_distance(z_mean).value

        # Convert template of z-shifts to chi-shifts
        if template_zmean_shifts is None:
            chimean_shifts_array = np.zeros(grid.npix, dtype=float)
        else:
            chimean_shifts_array = Planck18.comoving_distance(z_mean + template_zmean_shifts.map).value \
                                   - Planck18.comoving_distance(z_mean).value
        if template_width_shifts is None:
            width_shifts_array = np.zeros(grid.npix, dtype=float)
        else:
            width_shifts_array = Planck18.comoving_distance(z_mean + template_width_shifts.map).value \
                                 - Planck18.comoving_distance(z_mean).value

        if template_interloper_frac is not None:
            # If studying interlopers, ignore variations in main Gaussian
            print('Studying interlopers ONLY and ignoring any other anisotropy')
            # ToDo: allow for both interlopers and anisotropy in bulk of dndz
            self.interloper_chi_mean_fid = Planck18.comoving_distance(interloper_z_mean).value
            self.interloper_chi_sigma_fid = Planck18.comoving_distance(interloper_z_mean + interloper_sigma).value \
                                            - Planck18.comoving_distance(interloper_z_mean).value
            # TODO: Implement non-gaussian dndz
            dndz_main = (1 / (  (self.chi_sigma_fid) * np.sqrt(2 * np.pi))) * \
                             np.exp( -(grid.chi_array - self.chi_mean_fid ) ** 2 / ( 2 * (self.chi_sigma_fid) ** 2))
            dndz_interloper = (1 / ((self.interloper_chi_sigma_fid) * np.sqrt(2 * np.pi))) * \
                        np.exp(-(grid.chi_array - self.interloper_chi_mean_fid) ** 2
                               / (2 * (self.interloper_chi_sigma_fid) ** 2))
            dndz_perturbed = (1-template_interloper_frac.map[..., np.newaxis])*dndz_main \
                             + template_interloper_frac.map[..., np.newaxis]*dndz_interloper
        else:
            # In each pixel, calculate the perturbed dndz as a Gaussian in chi
            dndz_perturbed = (1 / ((self.chi_sigma_fid + width_shifts_array[..., np.newaxis]) * np.sqrt(2 * np.pi))) * np.exp(
                -(grid.chi_array - self.chi_mean_fid - chimean_shifts_array[..., np.newaxis]) ** 2 / (
                            2 * (self.chi_sigma_fid + width_shifts_array[..., np.newaxis]) ** 2))

        # Take the fiducial dndz to be the monopole of the perturbed dndz
        dndz_fid = np.mean(self.mask[..., None] * dndz_perturbed, axis=0)

        # Convert dndz to selection function
        phi_perturbed_array = (Planck18.H(grid.z_array[0, :]) / c).value * dndz_perturbed
        phi_fid_array = (Planck18.H(grid.z_array[0, :]) / c).value * dndz_fid

        # Normalize the selection function so that \int_{0}^{inf} dr phi(r) = 1
        phi_norm, error = quadrature(interp1d(grid.chi_array, phi_fid_array), grid.chi_min_int, grid.chi_max_int,
                                     tol=1e-20, miniter=3000)
        phi_perturbed_array *= phi_norm ** (-1)
        phi_fid_array *= phi_norm ** (-1)

        if get_delta_p:
            # Go on to extract the alms at each chi, and so on
            super().__init__(grid, phi_perturbed_array, phi_fid_array, self.mask)
        else:
            # Alternatively, when doing the cosmic shear calculation, we only need these two
            self.phi_perturbed_array = phi_perturbed_array
            self.phi_fid_array = phi_fid_array

class GalShear(Field):
    def __init__(self, grid, sigma, z_mean, template_zmean_shifts=None, template_width_shifts=None):
        """ Observed cosmic shear field subject to anisotropy in the (Gaussian) dN/dz of the source galaxies
            - Inputs:
                * grid = Instance of grid class containing numerical hyperparams
                * sigma = float. Standard deviation of the fiducial dN/dz
                * z_mean = float. Central redshift of the fiducial dN/dz
                * template_zmean_shifts = instance of templates.Template with shifts in the mean redshift of the dN/dz
                * template_width_shifts = instance of templates.Template with shifts in the width of the dN/dz
        """
        self.sigma = sigma
        self.z_mean = z_mean
        self.template_zmean_shifts = template_zmean_shifts
        self.template_width_shifts = template_width_shifts
        if template_zmean_shifts is not None:
            self.mask = template_zmean_shifts.mask
        else:
            self.mask = template_width_shifts.mask

        g_d = GalDelta(grid, sigma, z_mean, 'dummy', template_zmean_shifts, template_width_shifts, get_delta_p=False)
        phi_fid_array = g_d.phi_fid_array
        phi_perturbed_array = g_d.phi_perturbed_array
        self.chi_mean_fid = g_d.chi_mean_fid
        self.chi_sigma_fid = g_d.chi_sigma_fid

        # Calculate the galaxy lensing kernel
        # But first, a dummy run to pre-compile with numba
        utils.lens_efficiency_kernel(grid.chi_array, grid.chi_max_int, np.zeros_like(grid.chi_array))

        # Now in earnest
        g_fid = 3 / 2. * Planck18.Om0 * Planck18.H0.value ** 2 / c.value ** 2 * grid.chi_array * (
                1 + grid.z_array[0, :]) * utils.lens_efficiency_kernel(grid.chi_array, grid.chi_max_int, phi_fid_array)
        g_pert = 3 / 2. * Planck18.Om0 * Planck18.H0.value ** 2 / c.value ** 2 * grid.chi_array * (
                1 + grid.z_array[0, :]) * utils.lens_efficiency_kernel(grid.chi_array, grid.chi_max_int, phi_perturbed_array)

        # Go on to extract the alms at each chi, and so on
        super().__init__(grid, g_pert, g_fid, self.mask)