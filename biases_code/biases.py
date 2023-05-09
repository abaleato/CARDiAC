import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18, z_at_value
from astropy.constants import c
from scipy.ndimage.filters import gaussian_filter
import astropy.units as u
from scipy import interpolate
from scipy.interpolate import interp1d
from py3nj import wigner3j
from scipy.integrate import quadrature
import utils
import galaxy_ps
import multiprocessing
from functools import partial
import pickle

class experiment:
    def __init__(self, sigma, z_mean, sigma_zshift, sigma_zwidth, zmean_shifts_array, sigma_shifts_array, bvec,
                 z_min_int=0.005, z_max_int = None, nside_upsampling=128, plots_dir='', k=None, smoothing_factor = 0,
                 n_samples_of_chi=2**10):
        """ Initialise a cosmology and experimental charactierstics
            - Inputs:
                * sigma = float. Standard deviation of the fiducial dndz
                * z_mean = float. Central redshift of the fiducial dndz
                * sigma_zshift = float. Standard deviation of the shifts in the central redshift of the distribution
                * sigma_zwidth = float. Standard deviation of the variations in the width of the distribution
                * zmean_shifts_array = np.array of floats. A template of the shifts in the mean z of the dndz
                * sigma_shifts_array = np.array of floats. A template of the shifts in the width of the dndz
                * nside = float. Nside that sets the size of pixels over which dn/dz is constant
                * bvec = list containing [b1,    b2,    bs2,   bnabla2, SN] to be fed to Anzu to obtain Pgg
                * z_min_int (optional) = float. Minimum of range for the integrals over chi
                * z_max_int (optional) = float. z_max_int
                * n_samples_of_chi (optional) = int (a power of 2). Number of samples in chi
                * nside_upsampling (optional) = int (a power of 2). nside of the upsampled pixelization
                * plots_dir (optional) = str. Path to save any plots.
                * k (optional) = np array of floats. k at which to evaluate Pkgg. If None, k = np.logspace(-3,0,200)
                * smoothing_factor (optional)=float. Fraction of pixel width by which to smooth the injected anisotropy
                * n_samples_of_chi = int. A power of 2, # of samples in comoving distance
        """
        self.sigma = sigma
        self.z_mean = z_mean
        self.z_min_int = z_min_int
        if z_max_int is None:
            z_max_int = z_mean + 2
        self.z_max_int = z_max_int
        self.sigma_zshift = sigma_zshift
        self.sigma_zwidth = sigma_zwidth
        self.npix = len(zmean_shifts_array)
        self.nside = hp.get_nside(zmean_shifts_array)
        self.zmean_shifts_array = zmean_shifts_array
        self.sigma_shifts_array = sigma_shifts_array
        self.bvec = bvec
        self.plots_dir = plots_dir
        if k is None:
            k = np.logspace(-3,0,200)
        self.smoothing_factor = smoothing_factor

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

        # Get the redshift corresponding to these values of chi in the Planck18 cosmology
        self.z_array = np.zeros((self.npix, n_samples_of_chi))
        for i, chi in enumerate(self.chi_array):
            self.z_array[:, i] = z_at_value(Planck18.comoving_distance, chi * u.Mpc)

        # Convert template of z-shifts to chi-shifts
        chimean_shifts_array = Planck18.comoving_distance(z_mean + zmean_shifts_array).value \
                               - Planck18.comoving_distance(z_mean).value
        width_shifts_array = Planck18.comoving_distance(z_mean + sigma_shifts_array).value \
                               - Planck18.comoving_distance(z_mean).value

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
        self.phi_fid = interp1d(self.chi_array, self.phi_fid_array)

        # Extract the perturbation
        self.delta_p_maps = self.phi_perturbed_array - self.phi_fid_array

        # Let us up-sample the maps
        # If running on a laptop, don't go above nside_upsampling=128 (nside_upsampling = 256 already requires 20Gb of memory)
        delta_p_maps_upsampled = np.zeros((hp.nside2npix(nside_upsampling), n_samples_of_chi))
        for i in range(len(self.chi_array)):
            delta_p_maps_upsampled[:, i] = hp.ud_grade(self.delta_p_maps[:, i], nside_upsampling)

        delta_p_maps = delta_p_maps_upsampled

        # Get the variance of Dphi in each slice -- from this we can directly calculate mode-coupling biases at large l
        self.variance_at_distance_slice = np.var(self.delta_p_maps, axis=0, ddof=1)
        # The kernels in Limber integral when approximating the mode-coupling bias in the limit l>>1
        self.analytic_proj_kernel = interp1d(self.chi_array, self.variance_at_distance_slice/self.chi_array**2)
        self.f = (self.chi_array - self.chi_mean_fid) * self.sigma_chishift / self.chi_sigma_fid ** 2 / np.sqrt(8 * np.pi)
        self.analytic_kernel_toy_model = interp1d(self.chi_array, self.f ** 2 * self.phi_fid_array ** 2 /self.chi_array**2)

        # To avoid ringing due to the hard edges on which we seed the anisotropy, we smooth the maps with a Gaussian
        # with sigma equal to 1/2 of the typical width one of the big pixels (characterized by nside, not nside_upsampling)
        # NOTE: this seems to be a bad idea, so default is to not smooth
        sigma_gaussian_smoothing = self.smoothing_factor * np.sqrt(4 * np.pi / self.npix) * (360 * 60 / (2 * np.pi))  # in arcmin
        beam = utils.bl(sigma_gaussian_smoothing, 3 * nside_upsampling - 1)

        # Take the spherical harmonic transform of each r slice. Conveniently, we can take complex SHT so array sizes reduce by x2
        delta_p_lm_of_chi = np.zeros((hp.Alm.getsize(3 * nside_upsampling - 1), n_samples_of_chi), dtype=complex)
        for i in range(n_samples_of_chi):
            delta_p_lm_of_chi[:, i] = hp.map2alm(delta_p_maps[:, i])
            # Smooth the map to reduce ringing due to the hard edges of the big pixels
            delta_p_lm_of_chi[:, i] = hp.almxfl(delta_p_lm_of_chi[:, i], beam)

        self.lmax = hp.Alm.getlmax(delta_p_lm_of_chi.shape[0])
        self.Cl_deltap_of_chi1_chi2 = np.zeros((self.lmax + 1, n_samples_of_chi, n_samples_of_chi))

        for i in range(delta_p_lm_of_chi.shape[0]):
            if i % 1000 == 0:
                # Print progress
                print('Completed {}%'.format(round(100 * i / delta_p_lm_of_chi.shape[0]), 3))
            l, m = hp.Alm.getlm(self.lmax, i)  # Get the l corresponding to each value of m
            if m != 0:
                # Healpix indexes only m (not -m, since Ylm=Yl-m for a real field), so correct the sum for this
                factor = 2
            else:
                factor = 1
            # Get angular PS and deconvolve pixel window function for all possible combinations of chi1 and chi2
            self.Cl_deltap_of_chi1_chi2[l, :, :] += factor * np.outer(delta_p_lm_of_chi[i, :],
                                                                 np.conj(delta_p_lm_of_chi[i, :])).real / (2 * l + 1) 

        # We'll need the interpolated version of this
        self.cldp_interp = interp1d(self.chi_array, np.diagonal(self.Cl_deltap_of_chi1_chi2, axis1=1, axis2=2), axis=-1)

        # Get the galaxy power spectrum for this sample
        # Evaluate predictions at the Planck 18 cosmology and redshifts within 7sigma of the dndzmean
        zs_sampled = np.linspace(z_mean - 7 * sigma, z_mean + 7 * sigma, 15)
        chis_sampled = Planck18.comoving_distance(zs_sampled).value
        self.Pkgg = galaxy_ps.get_galaxy_ps(bvec, k, zs_sampled)
        # Interpolate
        # Note: scipy.interp2d is deprecated, but it is MUCH faster than the new alternatives...
        self.Pkgg_interp = interpolate.interp2d(chis_sampled, k, self.Pkgg,
                                                          kind='linear', bounds_error=False, fill_value=0)

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

    def plot_DeltaPhi_variance(self):
        '''
        Plot the perturbation variance as a function of distance
        '''
        plt.axhline(0, color='gray', lw=0.5)
        plt.axvline(self.chi_mean_fid, color='k', ls='--')
        plt.plot(self.chi_array, self.variance_at_distance_slice)
        plt.ylabel(r'Var[$\Delta \phi(\chi)$] [Mpc$^{-2}$]')
        plt.xlabel(r'$\chi$ [Mpc]')
        plt.show()

    def plot_ClDphi_of_chi(self):
        '''
        Plot C_l^{\Delta \phi}(\chi,\chi) vs \mathrm{log}_{10}\,\chi for various l's
        '''
        for l_to_plot in np.linspace(10, self.lmax, 8, dtype=int):
            plt.plot(self.chi_array, np.diagonal(self.Cl_deltap_of_chi1_chi2, axis1=1, axis2=2)[l_to_plot, :],
                     label=r'$l={}$'.format(l_to_plot))
        plt.axvline(self.chi_mean_fid, ls='--', color='k', label=r'$r=\chi(z_{\mathrm{mean}})$')
        plt.xlabel(r'$\mathrm{log}_{10}\,\chi$')
        plt.ylabel(r'$C_l^{\Delta \phi}(\chi,\chi)$')
        plt.legend()
        plt.xlim([1, 5000])
        plt.show()

    def plot_ClDphi_of_chi_2D(self):
        '''
        Plot log C_l^{\Delta \phi}(\chi,\chi) against l and  \mathrm{log}_{10}\,\chi
        '''
        X, Y = np.meshgrid(np.arange(len(self.chi_array)), np.arange(self.lmax + 1))
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
        ax.set_ylim([0, self.lmax])

        plt.xlabel(r'$\mathrm{log}_{10}\,\chi$')
        plt.ylabel(r'$l$')

        label_locs = np.arange(1000, 7000, 1000, dtype=int)
        ax.set_xticks(utils.find_closest_indices(self.chi_array, label_locs))
        ax.set_xticklabels(label_locs.astype('str'))
        plt.legend()

        plt.title(r'$\mathrm{log}_{10} \, C_l^{\Delta \phi}(\chi)$')
        plt.colorbar(location='right')
        plt.show()

    def plot_ClDphi_of_chi_chiprime_2D(self):
        '''
        Plot log |C_l^{\Delta \phi}(\chi,\chi')| against l and  \mathrm{log}_{10}\,\chi
        '''
        chi_idx = np.where(np.log10(self.chi_array) > 3.2)[0][0]
        X, Y = np.meshgrid(np.arange(len(self.chi_array)), np.arange(self.lmax + 1))

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

        ax.set_ylim([0, self.lmax])

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

def mode_coupling_bias(exp, ells, lprime_max='none', num_processes=1, miniter=1000, maxiter=2000, tol=1e-12, mode='full'):
    """ Calculate the mode-coupling bias to the galaxy clustering power spectrum in the Limber approximation
        - Inputs:
            * exp = an instance of the experiment class
            * ells = np array of ints. The ells at which to evaluate this bias
            * lprime_max (optional) = int. Value of l above which we ignore the anisotropy in C_l^{\Delta \phi}
            * num_processes (optional) = int. Number of processes to use. If >1, use multirocessing
            * miniter (optional) = int. Minimum number of iterations for quadrature.
            * maxiter (optional) = int. Maximum number of iterations for quadrature.
            * tol (optional) = int. Error tolerance before breaking numerical integration.
            * mode (optional) = str. If 'full' do the full calc (still assumes Limber).
                                    If 'analytic_via_variance' use analytic kernel \mathrm{Var}[\phi](\chi)
                                    If 'analytic_toy_model' use analytic kernel [f(\chi)\bar{\phi}(\chi)]^2
    """
    if mode=='full':
        # Do the full calculation, using no analytic approximations
        integral_at_l = mode_coupling_bias_at_l
        if lprime_max=='none':
            lprime_max = exp.Cl_deltap_of_chi1_chi2.shape[0]-1
    elif mode=='analytic_via_variance':
        # Use the analytic approximation where the Limber kernel is \mathrm{Var}[\phi](\chi)
        exp.mc_kernel = exp.analytic_proj_kernel
        integral_at_l = analytic_mode_coupling_bias_at_l
    elif mode=='analytic_toy_model':
        # Use the analytic approximation where the Limber kernel is [f(\chi)\bar{\phi}(\chi)]^2
        exp.mc_kernel = exp.analytic_kernel_toy_model
        integral_at_l = analytic_mode_coupling_bias_at_l
    print('Using mode {}'.format(mode))

    if num_processes>1:
        # Use multiprocessing to speed up calculation
        print('Running in parallel with {} processes'.format(num_processes))
        pool = multiprocessing.Pool(num_processes)
        # Helper function (pool.map can only take one, iterable input)
        func = partial(integral_at_l, exp, lprime_max, miniter, maxiter, tol)
        conv_bias = np.array(pool.map(func, ells))
        pool.close()
    else:
        conv_bias = np.zeros_like(ells, dtype=float)
        for i, l in enumerate(ells):
            conv_bias[i] = integral_at_l(exp, lprime_max, miniter, maxiter, tol, l)
    return conv_bias

def mode_coupling_bias_at_l(exp, lprime_max, miniter, maxiter, tol, l):
    """ Calculate the mode-coupling bias to the galaxy clustering power spectrum in the Limber approximation,
        at a specific l.
        - Inputs:
            * exp = an instance of the experiment class
            * lprime_max = int. Value of l above which we ignore the anisotropy in C_l^{\Delta \phi}
            * miniter (optional) = int. Minimum number of iterations for quadrature.
            * maxiter (optional) = int. Maximum number of iterations for quadrature.
            * tol (optional) = int. Error tolerance before breaking numerical integration.
            * l = int. The multipole of \Delta C_l
    """
    print('Working on l={}'.format(l))
    integrand = np.zeros_like(exp.chi_array)
    cldp_interp = exp.cldp_interp(exp.chi_array)
    for L in range(l + lprime_max + 1):
        abslmL = np.abs(l - L)
        lpL = l + L
        if abslmL <= lprime_max:
            # Only loop if you will probe scales below cut
            Pk_interp = np.diagonal(np.flipud(exp.Pkgg_interp(exp.chi_array, (L + 0.5) / exp.chi_array)))
            for lprime in np.arange(abslmL, min(lprime_max, lpL) + 1, 1):
                if (l + lprime + L) % 2 == 0:
                    w3 = wigner3j(2*l, 2*L, 2*lprime, 0, 0, 0)
                    prefactor = w3 ** 2 * (2 * lprime + 1) * (2 * L + 1) / (4 * np.pi)
                    integrand += prefactor / exp.chi_array ** 2 * Pk_interp * cldp_interp[lprime]
    f = interp1d(exp.chi_array, integrand)
    result, error = quadrature(f, exp.chi_min_int, exp.chi_max_int, miniter=miniter, maxiter=maxiter, tol=tol)
    return result

def analytic_mode_coupling_bias_at_l(exp, dummy, miniter, maxiter, tol, l):
    """ Calculate the mode-coupling bias to the galaxy clustering power spectrum in the Limber approximation,
        at a specific l (l \gg 1 for this approach to be valid), using the variance of \Delta Phi in each chi slice
        - Inputs:
            * exp = an instance of the experiment class
            * dummy = whatever. We don't actually need this, but we have them in order to match the code structure
            * miniter (optional) = int. Minimum number of iterations for quadrature.
            * maxiter (optional) = int. Maximum number of iterations for quadrature.
            * tol (optional) = int. Error tolerance before breaking numerical integration.
            * l = int. The multipole of \Delta C_l
    """
    print('Working on l={}'.format(l))
    # Interpolate at the scales required by Limber
    Pkgg_interp_1D = interp1d(exp.chi_array, np.diagonal(np.flipud(exp.Pkgg_interp(exp.chi_array,
                                                                                   (l + 0.5) / exp.chi_array))))
    result, error = quadrature(limber_integral, exp.chi_min_int,
                                             exp.chi_max_int, args=(Pkgg_interp_1D, exp.mc_kernel),
                                             miniter=miniter, maxiter=maxiter, tol=tol)
    return result

def limber_integral(chi, Pkgg_interp_1D, kernel):
    return kernel(chi) * Pkgg_interp_1D(chi)

def additive_bias(exp, ells, num_processes=1, miniter=1000, maxiter=2000, tol=1e-12):
    """ Calculate the mode-coupling bias to the galaxy clustering power spectrum
        - Inputs:
            * exp = an instance of the experiment class
            * ells = np array of ints. The ells at which to evaluate this bias
            * num_processes (optional) = int. Number of processes to use. If >1, use multirocessing
            * miniter (optional) = int. Minimum number of iterations for quadrature.
            * maxiter (optional) = int. Maximum number of iterations for quadrature.
            * tol (optional) = int. Error tolerance before breaking numerical integration.
    """
    if num_processes>1:
        # Use multiprocessing to speed up calculation
        print('Running in parallel with {} processes'.format(num_processes))
        pool = multiprocessing.Pool(num_processes)
        # Helper function (pool.map can only take one, iterable input)
        func = partial(additive_bias_at_l, exp, miniter, maxiter, tol)
        additive_bias = np.array(pool.map(func, ells))
        pool.close()
    else:
        additive_bias = np.zeros_like(ells, dtype=float)
        for i, l in enumerate(ells):
            additive_bias[i] = additive_bias_at_l(exp, miniter, maxiter, tol, l)
    return additive_bias

def additive_bias_at_l(exp, miniter, maxiter, tol, l):
    """ Calculate the mode-coupling bias to the galaxy clustering power spectrum
        at a specific l.
        - Inputs:
            * exp = an instance of the experiment class
            * miniter (optional) = int. Minimum number of iterations for quadrature.
            * maxiter (optional) = int. Maximum number of iterations for quadrature.
            * tol (optional) = int. Error tolerance before breaking numerical integration.
            * l = int. The multipole of \Delta C_l
    """
    print('Working on l={}'.format(l))
    try:
        Clchi1chi2_interp = interpolate.RegularGridInterpolator((exp.chi_array, exp.chi_array),
                                                                exp.Cl_deltap_of_chi1_chi2[l, :, :],
                                                                method='linear', bounds_error=True, fill_value=0)

        result, error = quadrature(integrand_additive_term, exp.chi_min_int,
                                             exp.chi_max_int, args=(Clchi1chi2_interp, exp.chi_min_int, exp.chi_max_int),
                                             miniter=miniter, maxiter=maxiter, tol=tol)
    except IndexError as e:
        print(f"{e}")
        print('Setting the result to 0 because ClDeltaphi ought to be 0 at this l')
        result = 0
    return result

def unbiased_term(exp, ells, num_processes=1, miniter=1000, maxiter=2000, tol=1e-12):
    """ Calculate the unbiased contribution to the galaxy clustering power spectrum in the Limber approximation
        - Inputs:
            * exp = an instance of the experiment class
            * ells = np array of ints. The ells at which to evaluate this bias
            * num_processes (optional) = int. Number of processes to use. If >1, use multirocessing
            * miniter (optional) = int. Minimum number of iterations for quadrature.
            * maxiter (optional) = int. Maximum number of iterations for quadrature.
            * tol (optional) = int. Error tolerance before breaking numerical integration.
    """
    if num_processes>1:
        # Use multiprocessing to speed up calculation
        pool = multiprocessing.Pool(num_processes)
        print('Running in parallel with {} processes'.format(num_processes))
        # Helper function (pool.map can only take one, iterable input)
        func = partial(unbiased_term_at_l, exp, miniter, maxiter, tol)
        clgg_unbiased = np.array(pool.map(func, ells))
        pool.close()
    else:
        clgg_unbiased = np.zeros_like(ells, dtype=float)
        for i, l in enumerate(ells):
            clgg_unbiased[i] = unbiased_term_at_l(exp, miniter, maxiter, tol, l)
    return clgg_unbiased

def unbiased_term_at_l(exp, miniter, maxiter, tol, l):
    """ Calculate the unbiased contribution to the galaxy clustering power spectrum in the Limber approximation
        at a specific l.
        - Inputs:
            * exp = an instance of the experiment class
            * miniter (optional) = int. Minimum number of iterations for quadrature.
            * maxiter (optional) = int. Maximum number of iterations for quadrature.
            * tol (optional) = int. Error tolerance before breaking numerical integration.
            * l = int. The multipole of \Delta C_l
    """
    print('Working on l={}'.format(l))
    Pkgg_interp_1D = interp1d(exp.chi_array, np.diagonal(np.flipud(exp.Pkgg_interp(exp.chi_array,
                                                                                   (l + 0.5) / exp.chi_array))))
    result, error = quadrature(integrand_unbiased_auto_term, exp.chi_min_int, exp.chi_max_int,
                                         args=(exp.phi_fid, Pkgg_interp_1D), miniter=miniter, maxiter=maxiter, tol=tol)
    return result

#
# Loading an experiment from file
#

class dummy_class(object):
    '''
    A dummy class to initialize an empty object and populate it with the desired dictionary loaded from file
    '''
    def __init__(self):
        pass

    def save_properties(self, output_filename='./dict_with_properties'):
        """
        Save the dictionary of key properties to file
        Inputs:
            * output_filename = str. Output filename
        """
        with open(output_filename+'.pkl', 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

def load_from_file(filename='./dict_with_properties', verbose=False):
    """
    Load a dictionary of the key properties. Must have previously been save  experiment.save_properties()
    Inputs:
        * filename = str. Filename for the pickle object to be loaded
    Returns:
        * A dummy object with the right attributes
    """
    with open(filename+'.pkl', 'rb') as input:
        experiment_dict = pickle.load(input)
    if verbose:
        print('Successfully loaded experiment object with properties:\n')
        print(experiment_dict)

    dummy_object = dummy_class()
    dummy_object.__dict__ = experiment_dict
    return dummy_object
