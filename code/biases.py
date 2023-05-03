import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18, z_at_value
from astropy.constants import c
from scipy.ndimage.filters import gaussian_filter
import astropy.units as u
from scipy import interpolate
from scipy.interpolate import interp1d
from sympy.physics.wigner import wigner_3j
from scipy.integrate import quadrature
import utils
import galaxy_ps
import multiprocessing
from functools import partial

class experiment:
    def __init__(self, sigma, z_mean, sigma_zshift, sigma_zwidth, nside, bvec, z_min_int=0.005, z_max_int = None,
                 modulation_of_mean_of_draws=0, n_samples_of_chi=2**10, nside_upsampling=64, plots_dir='', k=None):
        """ Initialise a cosmology and experimental charactierstics
            - Inputs:
                * sigma = float. Standard deviation of the fiducial dndz
                * z_mean = float. Central redshift of the fiducial dndz
                * sigma_zshift = float. Standard deviation of the shifts in the central redshift of the distribution
                * sigma_zwidth = float. Standard deviation of the variations in the width of the distribution
                * nside = float. Nside that sets the size of pixels over which dn/dz is constant
                * bvec = list containing [b1,    b2,    bs2,   bnabla2, SN] to be fed to Anzu to obtain Pgg
                * z_min_int (optional) = float. Minimum of range for the integrals over chi
                * z_max_int (optional) = float. z_max_int
                * modulation_of_mean_of_draws (optional) = float. Modulation of the mean of these shifts across the sky
                * n_samples_of_chi (optional) = int (a power of 2). Number of samples in chi
                * nside_upsampling (optional) = int (a power of 2). nside of the upsampled pixelization
                * plots_dir (optional) = str. Path to save any plots.
                * k (optional) = np array of floats. k at which to evaluate Pkgg. If None, k = np.logspace(-3,0,200)
        """
        self.sigma = sigma
        self.z_mean = z_mean
        self.z_min_int = z_min_int
        if z_max_int is None:
            z_max_int = z_mean + 2
        self.z_max_int = z_max_int
        self.sigma_zshift = sigma_zshift
        self.sigma_zwidth = sigma_zwidth
        self.nside = nside
        self.bvec = bvec
        self.plots_dir = plots_dir
        if k is None:
            k = np.logspace(-3,0,200)
        self.npix = hp.nside2npix(nside)

        # The user input is in redshift units because this is more intuitive. However, we will define our dndzs to be
        # Gaussian in comoving distance. So next, we convert to chi
        self.chi_mean_fid = Planck18.comoving_distance(z_mean).value
        # Width of the fiducial distribution
        self.chi_sigma_fid = Planck18.comoving_distance(z_mean + sigma).value - Planck18.comoving_distance(z_mean).value
        # Integral ranges
        self.chi_min_int = Planck18.comoving_distance(z_min_int).value
        self.chi_max_int = Planck18.comoving_distance(z_max_int).value
        self.chi_array = np.linspace(self.chi_min_int, self.chi_max_int + 100, n_samples_of_chi)

        # Sigma of the distributions from which to draw shifts
        self.sigma_chishift = Planck18.comoving_distance(z_mean + sigma_zshift).value - Planck18.comoving_distance(
            z_mean).value
        self.sigma_chiwidth = Planck18.comoving_distance(z_mean + sigma_zwidth).value - Planck18.comoving_distance(
            z_mean).value

        # Initialize samples in chi (chi is comoving distance throughout)
        n_samples_of_chi = 2 ** 10  # Choose a power of 2

        # Get the redshift corresponding to these values of chi in the Planck18 cosmology
        self.z_array = np.zeros((self.npix, n_samples_of_chi))
        for i, chi in enumerate(self.chi_array):
            self.z_array[:, i] = z_at_value(Planck18.comoving_distance, chi * u.Mpc)

        # Draw random values in each pixel for the shift of the central redshift of the dndz
        chimean_shifts_array = np.random.normal(loc=modulation_of_mean_of_draws, scale=self.sigma_chishift, size=self.npix)
        width_shifts_array = np.random.normal(loc=modulation_of_mean_of_draws, scale=self.sigma_chiwidth, size=self.npix)

        # In each pixel, calculate the perturbed dndz as a Gaussian in chi
        dndz_perturbed = (1 / ((self.chi_sigma_fid + width_shifts_array[..., np.newaxis]) * np.sqrt(2 * np.pi))) * np.exp(
            -(self.chi_array - self.chi_mean_fid - chimean_shifts_array[..., np.newaxis]) ** 2 / (
                        2 * (self.chi_sigma_fid + width_shifts_array[..., np.newaxis]) ** 2))
        # Take the fiducial dndz to be the monopole of the perturbed dndz
        dndz_fid = np.mean(dndz_perturbed, axis=0)

        # Convert dndz to selection function
        self.phi_perturbed_array = (Planck18.H(self.z_array[0, :]) / c).value * dndz_perturbed
        self.phi_fid_array = (Planck18.H(self.z_array[0, :]) / c).value * dndz_fid

        # Normalize the selection function so that \int_{0}^{inf} dr phi(r) = 1
        phi_norm, error = quadrature(interp1d(self.chi_array, self.phi_fid_array), self.chi_min_int, self.chi_max_int,
                                     tol=1e-20, miniter=3000)
        self.phi_perturbed_array *= phi_norm ** (-1)
        self.phi_fid_array *= phi_norm ** (-1)
        dndz_fid_normed = dndz_fid / phi_norm
        phi_fid = interp1d(self.chi_array, self.phi_fid_array)

        # Extract the perturbation
        self.delta_p_maps = self.phi_perturbed_array - self.phi_fid_array

        # Let us up-sample the maps
        # If running on a laptop, don't go above nside_upsampling=128 (nside_upsampling = 256 already requires 20Gb of memory)
        delta_p_maps_upsampled = np.zeros((hp.nside2npix(nside_upsampling), n_samples_of_chi))
        for i in range(len(self.chi_array)):
            delta_p_maps_upsampled[:, i] = hp.ud_grade(self.delta_p_maps[:, i], nside_upsampling)

        delta_p_maps = delta_p_maps_upsampled

        # To avoid ringing due to the hard edges on which we seed the anisotropy, we smooth the maps with a Gaussian
        # with sigma equal to 1/2 of the typical width one of the big pixels (characterized by nside, not nside_upsampling)
        sigma_gaussian_smoothing = np.sqrt(4 * np.pi / self.npix) / 2. * (360 * 60 / (2 * np.pi))  # in arcmin

        def bl(fwhm_arcmin, lmax):
            """ returns the map-level transfer function for a symmetric Gaussian beam.
                 * fwhm_arcmin      = beam full-width-at-half-maximum (fwhm) in arcmin.
                 * lmax             = maximum multipole.
            """
            ls = np.arange(0, lmax + 1)
            return np.exp(-(fwhm_arcmin * np.pi / 180. / 60.) ** 2 / (16. * np.log(2.)) * ls * (ls + 1.))

        beam = bl(sigma_gaussian_smoothing, 3 * nside_upsampling - 1)

        # Take the spherical harmonic transform of each r slice. Conveniently, we can take complex SHT so array sizes reduce by x2
        delta_p_lm_of_chi = np.zeros((hp.Alm.getsize(3 * nside_upsampling - 1), n_samples_of_chi), dtype=complex)
        for i in range(n_samples_of_chi):
            delta_p_lm_of_chi[:, i] = hp.map2alm(delta_p_maps[:, i])
            # Smooth the map to reduce ringing due to the hard edges of the big pixels
            delta_p_lm_of_chi[:, i] = hp.almxfl(delta_p_lm_of_chi[:, i], beam)

        lmax = hp.Alm.getlmax(delta_p_lm_of_chi.shape[0])
        self.Cl_deltap_of_chi1_chi2 = np.zeros((lmax + 1, n_samples_of_chi, n_samples_of_chi))
        pixwinf = hp.pixwin(nside_upsampling)[0:lmax + 1]  # Get the pixel window function for the up-sampled pixelization

        for i in range(delta_p_lm_of_chi.shape[0]):
            if i % 1000 == 0:
                # Print progress
                print('Completed {}%'.format(round(100 * i / delta_p_lm_of_chi.shape[0]), 3))
            l, m = hp.Alm.getlm(lmax, i)  # Get the l corresponding to each value of m
            if m != 0:
                # Healpix indexes only m (not -m, since Ylm=Yl-m for a real field), so correct the sum for this
                factor = 2
            else:
                factor = 1
            # Get angular PS and deconvolve pixel window function for all possible combinations of chi1 and chi2
            self.Cl_deltap_of_chi1_chi2[l, :, :] += factor * np.outer(delta_p_lm_of_chi[i, :],
                                                                 np.conj(delta_p_lm_of_chi[i, :])).real / (2 * l + 1) / \
                                               pixwinf[l] ** 2

        # We'll need the interpolated version of this
        self.cldp_interp = interp1d(self.chi_array, np.diagonal(self.Cl_deltap_of_chi1_chi2, axis1=1, axis2=2), axis=-1)

        # Get the galaxy power spectrum for this sample
        # Evaluate predictions at the Planck 18 cosmology and redshifts within 7sigma of the dndzmean
        zs_sampled = np.linspace(z_mean - 7 * sigma, z_mean + 7 * sigma, 15)
        chis_sampled = Planck18.comoving_distance(zs_sampled).value
        self.Pkgg = galaxy_ps.get_galaxy_ps(bvec, k, zs_sampled)
        # Interpolate
        self.Pkgg_interp = interpolate.RegularGridInterpolator((k, chis_sampled), self.Pkgg,
                                                          method='linear', bounds_error=False, fill_value=0)

    def save_properties(self, output_filename='./dict_with_properties'):
        """
        Save the dictionary of key properties to file
        Inputs:
            * output_filename = str. Output filename
        """
        with open(output_filename+'.pkl', 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def plot_realizations(self):
        '''
        Plot some typical realizations
        '''
        for pixel_id in np.random.randint(0, self.npix, 5):
            plt.plot(self.z_array[0, :], self.phi_perturbed_array[pixel_id, :], label='Actual, pixel {}'.format(pixel_id))

        # Plot the fiducial dndz
        plt.plot(self.z_array[0, :], self.phi_fid_array, color='k', lw=3, label='fiducial')

        plt.ylabel(r'$\phi(z)$')
        plt.xlabel(r'$z$')
        plt.xlim([0, 2])
        plt.legend()
        plt.show()

    def plot_fluctuations(self):
        '''
        Plot a map of an chi-slice where we expect significant fluctuations
        '''
        chi_mean_idx = np.where(self.chi_array > self.chi_mean_fid)[0][0]
        hp.mollview(self.delta_p_maps[:, chi_mean_idx])
        plt.show()

        # Plot the Delta phi (z) at a random pixel
        pix_idx = np.random.randint(0, self.npix)
        plt.plot(self.z_array[pix_idx, :], self.delta_p_maps[pix_idx, :])
        plt.ylabel(r'$\Delta \phi(z)$')
        plt.xlabel(r'$z$')
        plt.show()

    def plot_ClDphi_of_chi(self, lmax=300):
        '''
        Plot C_l^{\Delta \phi}(\chi,\chi) vs \mathrm{log}_{10}\,\chi for various l's
        '''
        for l_to_plot in np.linspace(10, lmax, 8, dtype=int):
            plt.plot(self.chi_array, np.diagonal(self.Cl_deltap_of_chi1_chi2, axis1=1, axis2=2)[l_to_plot, :],
                     label=r'$l={}$'.format(l_to_plot))
        plt.axvline(self.chi_mean_fid, ls='--', color='k', label=r'$r=\chi(z_{\mathrm{mean}})$')
        plt.xlabel(r'$\mathrm{log}_{10}\,\chi$')
        plt.ylabel(r'$C_l^{\Delta \phi}(\chi,\chi)$')
        plt.legend()
        plt.xlim([1, 5000])
        plt.show()

    def plot_ClDphi_of_chi_2D(self, lmax=300):
        '''
        Plot log C_l^{\Delta \phi}(\chi,\chi) against l and  \mathrm{log}_{10}\,\chi
        '''
        X, Y = np.meshgrid(np.arange(len(self.chi_array)), np.arange(lmax + 1))
        Z = np.log10(np.diagonal(self.Cl_deltap_of_chi1_chi2, axis1=1, axis2=2))
        # Set the log of 0 to a tiny negative number
        Z[np.diagonal(self.Cl_deltap_of_chi1_chi2, axis1=1, axis2=2) == 0] = -1e100

        contours = np.linspace(-20, -9, 20)

        # Smooth array with a Gaussian filter for plotting purposes
        Z_smoothed = gaussian_filter(Z, sigma=3)
        plt.contourf(X, Y, Z_smoothed, levels=contours, cmap='inferno', extend='both')

        ax = plt.gca()
        ax.axvline(np.where(self.chi_array > self.chi_mean_fid)[0][0], color='r', ls='--', lw=1,
                   label=r'$r=\chi(z_{\mathrm{mean}})$')
        ax.set_ylim([0, lmax])

        plt.xlabel(r'$\mathrm{log}_{10}\,\chi$')
        plt.ylabel(r'$l$')

        label_locs = np.arange(1000, 7000, 1000, dtype=int)
        ax.set_xticks(utils.find_closest_indices(self.chi_array, label_locs))
        ax.set_xticklabels(label_locs.astype('str'))
        plt.legend()

        plt.title(r'$\mathrm{log}_{10} \, C_l^{\Delta \phi}(\chi)$')
        plt.colorbar(location='right')
        plt.show()

    def plot_ClDphi_of_chi_chiprime_2D(self, lmax=300):
        '''
        Plot log |C_l^{\Delta \phi}(\chi,\chi')| against l and  \mathrm{log}_{10}\,\chi
        '''
        chi_idx = np.where(np.log10(self.chi_array) > 3.2)[0][0]
        X, Y = np.meshgrid(np.arange(len(self.chi_array)), np.arange(lmax + 1))

        Z_fixed_chi = np.log10(np.abs(self.Cl_deltap_of_chi1_chi2[:, :, chi_idx]))
        # Set the log of 0 to a tiny negative number
        Z_fixed_chi[np.abs(self.Cl_deltap_of_chi1_chi2[:, :, chi_idx]) == 0] = -1e100

        # Smooth array with a Gaussian filter for plotting purposes
        Z_fixed_chi_smoothed = gaussian_filter(Z_fixed_chi, sigma=3)

        contours = np.linspace(-20, -9, 20)
        plt.contourf(X, Y, Z_fixed_chi_smoothed, levels=contours, cmap='inferno', extend='both')

        plt.xlabel(r'$\mathrm{log}_{10}\,\chi$')
        plt.ylabel(r'$l$')

        ax = plt.gca()
        ax.axvline(np.where(self.chi_array > self.chi_mean_fid)[0][0], color='r', ls='--', lw=1,
                   label=r'$r=r(z_{\mathrm{mean}})$')

        ax.set_ylim([0, lmax])

        label_locs = np.arange(1000, 7000, 1000, dtype=int)
        ax.set_xticks(utils.find_closest_indices(self.chi_array, label_locs))
        ax.set_xticklabels(label_locs.astype('str'))

        plt.title(r'$\mathrm{log}_{10} \, |C_l^{\Delta \phi}(\chi,\chi^{*})|$')
        plt.colorbar(location='right')

        ax.axvline(chi_idx, label=r'$\chi^{*}$', ls=':', color='w')
        plt.legend()
        plt.show()

    def plot_ClDphi_vs_chi_chiprime_2D(self, l):
        '''
        Plot C_l^{\Delta \phi}(\chi,\chi') against \chi \chi' a fixed l
        '''
        contours = np.linspace(-2.5e-9, 2.5e-9, 20)
        suffix = ''

        X, Y = np.meshgrid(np.arange(len(self.chi_array)), np.arange(len(self.chi_array)))
        Z = self.Cl_deltap_of_chi1_chi2[l, :, :]

        plt.contourf(X, Y, Z, cmap='RdBu', levels=contours, extend='neither')

        ax = plt.gca()
        ax.axvline(np.where(self.chi_array > self.chi_mean_fid)[0][0], color='k', ls=':', lw=1, label=r'$\chi=\chi_0$')
        ax.axhline(np.where(self.chi_array > self.chi_mean_fid)[0][0], color='k', ls=':', lw=1)

        plt.xlabel(r'$\chi_1$ [Mpc]')
        plt.ylabel(r'$\chi_2$ [Mpc]')

        label_locs = np.arange(1000, 7000, 1000, dtype=int)
        ax.set_xticks(utils.find_closest_indices(self.chi_array, label_locs))
        ax.set_yticks(utils.find_closest_indices(self.chi_array, label_locs))

        ax.set_xticklabels(label_locs.astype('str'))
        ax.set_yticklabels(label_locs.astype('str'))
        plt.legend()

        plt.title(r'$ C_\ell^{\Delta \phi \Delta \phi}(\chi_1, \chi_2)$ for $\ell=$' + str(l))
        cbar = plt.colorbar(location='right')
        cbar_labels = np.linspace(np.min(contours), np.max(contours), 5, dtype=float)
        cbar.set_ticks(cbar_labels)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)

        plt.xlim([np.where(self.chi_array > 900)[0][0], np.where(self.chi_array > 3500)[0][0]])
        plt.ylim([np.where(self.chi_array > 900)[0][0], np.where(self.chi_array > 3500)[0][0]])
        plt.show()

#
# Integrands
#

def integrand_conv_term(chi, small_l, Pkgg_interp_1Dlimber, prefactor, cldp_interp):
    '''
    Integrand for the convolutional bias term in the Limber approximation.
    Requires globally-defined cldp_interp(k)[l], interpolated in k
    Pkgg_interp_1Dlimber = interp1d(chi_array, Pkgg( (L+0.5)/chi_array, z(chi_array) ))
    '''
    return prefactor/ chi**2 * Pkgg_interp_1Dlimber(chi) * cldp_interp(chi)[small_l]

def integrand_additive_term(chi1, Clchi1chi2_interp, chi_min_int, chi_max_int):
    '''
    Integrand for the additive bias term, in the Limber approximation.
    Requires globally-defined cldp_interp(k1, k2)[l], interpolated in k
    '''
    outer_integral, error = quadrature(integrand_nested_additive_term, chi_min_int,
                                       chi_max_int, args=(chi1, Clchi1chi2_interp), miniter=3,
                                             maxiter=5, tol=1e-20)
    return outer_integral

def integrand_nested_additive_term(chi2, chi1, Clchi1chi2_interp):
    chi1=chi1[0]
    chi2=chi2[0]
    return Clchi1chi2_interp(np.array((chi1, chi2)))[0]

def integrand_unbiased_auto_term(chi, phi_fid, Pkgg_interp_1D):
    '''
    Integrand for the unbiased Clgg auto spectrum in the Limber approximation.
    Requires globally-defined Pkgg_interp_1D(chi)
    '''
    return Pkgg_interp_1D(chi) * (phi_fid(chi)/chi)**2

#
# Integral evaluations
#

def mode_coupling_bias(exp, ells, lprime_max=100, parallelize=False):
    """ Calculate the mode-coupling bias to the galaxy clustering power spectrum in the Limber approximation
        - Inputs:
            * exp = an instance of the experiment class
            * ells = np array of ints. The ells at which to evaluate this bias
            * lprime_max (optional) = int. Value of l above which we ignore the anisotropy in C_l^{\Delta \phi}
            * parallelize (optional) = Bool. Whether or not to use multiprocessing to evaluate
    """
    if parallelize:
        # Use multiprocessing to speed up calculation
        if len(ells)>multiprocessing.cpu_count():
            # Start as many processes as machine can handle
            num_processes = multiprocessing.cpu_count()
        else:
            # Only start as many as you need
            num_processes = len(ells)
        print('Running in parallel with {} processes'.format(num_processes))
        pool = multiprocessing.Pool(num_processes)
        # Helper function (pool.map can only take one, iterable input)
        func = partial(mode_coupling_bias_at_l, exp, lprime_max)
        conv_bias = np.array(pool.map(func, ells))
        pool.close()
    else:
        conv_bias = np.zeros_like(ells, dtype=float)
        for i, l in enumerate(ells):
            conv_bias[i] = mode_coupling_bias_at_l(exp, lprime_max, l)
    return conv_bias

def mode_coupling_bias_at_l(exp, lprime_max, l):
    """ Calculate the mode-coupling bias to the galaxy clustering power spectrum in the Limber approximation,
        at a specific l.
        - Inputs:
            * exp = an instance of the experiment class
            * lprime_max = int. Value of l above which we ignore the anisotropy in C_l^{\Delta \phi}
            * l = int. The multipole of \Delta C_l
    """
    print('Working on l={}'.format(l))
    result = 0
    for lprime in range(lprime_max):
        for L in np.arange(np.abs(l - lprime), np.abs(l + lprime) + 1, 1):
            if (l + lprime + L) % 2 == 0:
                w3 = wigner_3j(l, L, lprime, 0, 0, 0)
                prefactor = float(w3) ** 2 * (2 * lprime + 1) * (2 * L + 1) / (4 * np.pi)

                # Interpolate at the scales required by Limber
                X, Y = np.meshgrid((L + 0.5) / exp.chi_array, exp.chi_array, indexing='ij')
                Pkgg_interp_1D = interp1d(exp.chi_array, np.diagonal(exp.Pkgg_interp((X, Y))))

                integ, error = quadrature(integrand_conv_term, exp.chi_min_int, exp.chi_max_int,
                                          args=(lprime, Pkgg_interp_1D, prefactor, exp.cldp_interp), miniter=1000,
                                          maxiter=2000,
                                          tol=1e-12)
                result += integ
    return result

def additive_bias(exp, ells, parallelize=False):
    """ Calculate the mode-coupling bias to the galaxy clustering power spectrum
        - Inputs:
            * exp = an instance of the experiment class
            * ells = np array of ints. The ells at which to evaluate this bias
            * parallelize (optional) = Bool. Whether or not to use multiprocessing to evaluate
    """
    if parallelize:
        # Use multiprocessing to speed up calculation
        if len(ells)>multiprocessing.cpu_count():
            # Start as many processes as machine can handle
            num_processes = multiprocessing.cpu_count()
        else:
            # Only start as many as you need
            num_processes = len(ells)
        print('Running in parallel with {} processes'.format(num_processes))
        # Helper function (pool.map can only take one, iterable input)
        func = partial(additive_bias_at_l, exp)
        additive_bias = np.array(pool.map(func, ells))
        pool.close()
    else:
        additive_bias = np.zeros_like(ells, dtype=float)
        for i, l in enumerate(ells):
            additive_bias[i], error = additive_bias_at_l(exp, l)
    return additive_bias

def additive_bias_at_l(exp, l):
    """ Calculate the mode-coupling bias to the galaxy clustering power spectrum
        at a specific l.
        - Inputs:
            * exp = an instance of the experiment class
            * l = int. The multipole of \Delta C_l
    """
    print('Working on l={}'.format(l))
    Clchi1chi2_interp = interpolate.RegularGridInterpolator((exp.chi_array, exp.chi_array),
                                                            exp.Cl_deltap_of_chi1_chi2[l, :, :],
                                                            method='linear', bounds_error=True, fill_value=0)

    result, error = quadrature(integrand_additive_term, exp.chi_min_int,
                                         exp.chi_max_int, args=(Clchi1chi2_interp, exp.chi_min_int, exp.chi_max_int),
                                         miniter=3, maxiter=5, tol=1e-20)
    return result

def unbiased_term(exp, ells, parallelize=False):
    """ Calculate the unbiased contribution to the galaxy clustering power spectrum in the Limber approximation
        - Inputs:
            * exp = an instance of the experiment class
            * ells = np array of ints. The ells at which to evaluate this bias
            * parallelize (optional) = Bool. Whether or not to use multiprocessing to evaluate
    """
    if parallelize:
        # Use multiprocessing to speed up calculation
        if len(ells)>multiprocessing.cpu_count():
            # Start as many processes as machine can handle
            num_processes = multiprocessing.cpu_count()
        else:
            # Only start as many as you need
            num_processes = len(ells)
        print('Running in parallel with {} processes'.format(num_processes))
        # Helper function (pool.map can only take one, iterable input)
        func = partial(unbiased_term_at_l, exp)
        clgg_unbiased = np.array(pool.map(func, ells))
        pool.close()
    else:
        clgg_unbiased = np.zeros_like(ells, dtype=float)
        for i, l in enumerate(ells):
            clgg_unbiased[i] = unbiased_term_at_l(exp, l)
    return clgg_unbiased

def unbiased_term_at_l(exp, l):
    """ Calculate the unbiased contribution to the galaxy clustering power spectrum in the Limber approximation
        at a specific l.
        - Inputs:
            * exp = an instance of the experiment class
            * l = int. The multipole of \Delta C_l
    """
    print('Working on l={}'.format(l))
    X, Y = np.meshgrid((l + 0.5) / exp.chi_array, exp.chi_array, indexing='ij')
    Pkgg_interp_1D = interp1d(exp.chi_array, np.diagonal(exp.Pkgg_interp((X, Y))))
    result, error = quadrature(integrand_unbiased_auto_term, exp.chi_min_int, exp.chi_max_int,
                                         args=(Pkgg_interp_1D), miniter=1000, maxiter=2000, tol=1e-12)
    return result

def load_from_file(filename='./dict_with_biases.pkl'):
    """
    Load a dictionary of the key properties. Must have previously been save  experiment.save_properties()
    Inputs:
        * filename = str. Filename for the pickle object to be loaded
    Returns:
        * A dummy object with the right attributes
    """
    with open(filename, 'rb') as input:
        experiment_dict = pickle.load(input)
    print('Successfully loaded experiment object with properties:\n')

    class A(object):
        def __init__(self):
    dummy_object = A()
    dummy_object.__dict__ = experiment_dict
    return dummy_object
