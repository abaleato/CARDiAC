import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import utils
import _pickle as pickle

class Spec:
    def __init__(self, field1=None, field2=None, load=False, save=False, filename=None):
        """ General class defining a spectrum between two fields, including infrastruture to compute all contributions
            - Inputs:
                * field1 (optional) = Instance of fields.Field class. Needed when not loading from file
                * field2 (optional) = Instance of fields.Field class. Defaults to field1
                * load (optional) = Bool. Whether to load object from file. If True, filename must be provided
                * save (optional) = Bool. Whether to save object to file. If True, filename must be provided
                * filename (optional) = str. Path to pickled object to write to / load from
        """
        assert (field1 is not None or filename is not None), "Must either initialize new object or load one from file!"
        if (load!=False) or (save!=False):
            assert (filename is not None), "Must provide a filename when saving or loading"
        if field2 is None:
            field2 = field1
        assert(field1 == field2), "Fields are incompatible given their grids"

        if load:
            with open(filename + '.pkl', 'rb') as input:
                self.__dict__ = pickle.load(input)
        else:
            self.field1 = field1
            self.field2 = field2
            self.label_dict = {'GalDelta': r'\phi', 'GalShear': r'g'}
            self.grid = field1.grid

            # Get the (co)variance btw delta_p's in each slice
            # From this we can directly calculate mode-coupling biases at large l
            covmat = np.cov(field1.delta_p_maps, field2.delta_p_maps, ddof=1, rowvar=False)
            self.cov_at_chi = np.diagonal(covmat, offset=field1.grid.n_samples_of_chi)

            # The fiducial projection kernels
            self.kernel = interp1d(self.grid.chi_array, field1.p_fid_array * field2.p_fid_array)

            # The kernels in Limber integral when approximating the mode-coupling bias in the limit l>>1
            #self.analytic_proj_kernel = interp1d(self.chi_array, self.variance_at_distance_slice/self.chi_array**2)
        if save:
            with open(filename + '.pkl', 'wb') as output:
                pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)


    def get_Cldp1dp2(self):
        self.lmax = hp.Alm.getlmax(self.field1.delta_p_lm_of_chi.shape[0])
        self.Cl_deltap_of_chi1_chi2 = np.zeros((self.lmax + 1, self.field1.grid.n_samples_of_chi,
                                                self.field2.grid.n_samples_of_chi))
        for i in range(self.field1.delta_p_lm_of_chi.shape[0]):
            l, m = hp.Alm.getlm(self.lmax, i)  # Get the l corresponding to each value of m
            if m != 0:
                # Healpix indexes only m (not -m, since Ylm=Yl-m for a real field), so correct the sum for this
                factor = 2
            else:
                factor = 1
            # Get angular PS and deconvolve pixel window function for all possible combinations of chi1 and chi2
            self.Cl_deltap_of_chi1_chi2[l, :, :] += factor * np.outer(self.field1.delta_p_lm_of_chi[i, :],
                                                                      np.conj(
                                                                          self.field2.delta_p_lm_of_chi[i, :])).real / (
                                                            2 * l + 1)

        # We'll need the interpolated version of this
        self.cldp_interp = interp1d(self.field1.grid.chi_array,
                                    np.diagonal(self.Cl_deltap_of_chi1_chi2, axis1=1, axis2=2), axis=-1)

    '''
    def get_biases(self)

    def get_3D_spectrum(self):
        # Get the galaxy power spectrum for this sample
        # Evaluate predictions at the Planck 18 cosmology and redshifts within 7sigma of the dndzmean
        zs_sampled = np.linspace(z_mean - 7 * sigma, z_mean + 7 * sigma, 15)
        chis_sampled = Planck18.comoving_distance(zs_sampled).value
        self.Pkgg = galaxy_ps.get_galaxy_ps(bvec, k, zs_sampled)
        # Interpolate
        # Note: scipy.interp2d is deprecated, but it is MUCH faster than the new alternatives...
        self.Pkgg_interp = interpolate.interp2d(chis_sampled, k, self.Pkgg,
                                                kind='linear', bounds_error=False, fill_value=0)
    '''
    def plot_DeltaP_covariance(self):
        '''
        Plot the perturbation covariance as a function of distance
        '''
        plt.axhline(0, color='gray', lw=0.5)
        plt.axvline(self.field1.chi_mean_fid, color='k', ls='--')
        plt.plot(self.grid.chi_array, self.cov_at_chi)
        plt.ylabel(r'Cov[$\Delta '+self.label_dict[self.field1.__class__.__name__]+'(\chi), \Delta '+
                   self.label_dict[self.field2.__class__.__name__]+'(\chi)$] [Mpc$^{-2}$]')
        plt.xlabel(r'$\chi$ [Mpc]')
        plt.show()

    def plot_Cl_DeltaP_of_chi(self):
        '''
        Plot C_l^{\Delta \phi}(\chi,\chi) vs \mathrm{log}_{10}\,\chi for various l's
        '''
        for l_to_plot in np.linspace(10, self.lmax, 8, dtype=int):
            plt.plot(self.grid.chi_array, np.diagonal(self.Cl_deltap_of_chi1_chi2, axis1=1, axis2=2)[l_to_plot, :],
                     label=r'$l={}$'.format(l_to_plot))
        plt.axvline(self.field1.chi_mean_fid, ls='--', color='k', label=r'$r=\chi(z_{\mathrm{mean}})$')
        plt.xlabel(r'$\mathrm{log}_{10}\,\chi$')
        plt.ylabel(r'$C_l^{\Delta '+self.label_dict[self.field1.__class__.__name__]+'}(\chi,\chi)$')
        plt.legend()
        plt.xlim([1, self.field1.chi_mean_fid + 10*self.field1.chi_sigma_fid])
        plt.show()

    def plot_Cl_DeltaP_of_chi_2D(self, xmin=None, xmax=None, min_log_range=-20, max_log_range=-9):
        '''
        Plot log C_l^{\Delta \phi}(\chi,\chi) against l and  \mathrm{log}_{10}\,\chi
        '''
        if xmin is None:
            xmin = self.field1.chi_mean_fid - 7*self.field1.chi_sigma_fid
        if xmax is None:
            xmax = self.field1.chi_mean_fid + 7*self.field1.chi_sigma_fid

        X, Y = np.meshgrid(np.arange(len(self.grid.chi_array)), np.arange(self.lmax + 1))
        Z = np.log10(np.diagonal(self.Cl_deltap_of_chi1_chi2, axis1=1, axis2=2))
        # Set the log of 0 to a tiny negative number
        Z[np.diagonal(self.Cl_deltap_of_chi1_chi2, axis1=1, axis2=2) == 0] = -1e100

        contours = np.linspace(min_log_range, max_log_range, 20)

        # Smooth array with a Gaussian filter for plotting purposes
        Z_smoothed = gaussian_filter(Z, sigma=3)
        plt.contourf(X, Y, Z_smoothed, levels=contours, cmap='inferno', extend='both')

        ax = plt.gca()
        ax.axvline(np.where(self.grid.chi_array > self.field1.chi_mean_fid)[0][0], color='r', ls='--', lw=1,
                   label=r'$r=\chi(z_{\mathrm{mean}})$')
        ax.set_ylim([0, self.lmax])

        plt.xlabel(r'$\mathrm{log}_{10}\,\chi$')
        plt.ylabel(r'$l$')

        label_locs = np.round(np.linspace(xmin+50, xmax-50, 3, dtype=int), decimals=-1)
        ax.set_xticks(utils.find_closest_indices(self.grid.chi_array, label_locs))
        ax.set_xticklabels(label_locs.astype('str'))
        plt.legend()

        plt.title(r'$\mathrm{log}_{10} \, C_l^{\Delta '+self.label_dict[self.field1.__class__.__name__]+'}(\chi)$')
        plt.colorbar(location='right')
        plt.xlim([np.where(self.grid.chi_array > xmin)[0][0], np.where(self.grid.chi_array > xmax)[0][0]])
        plt.show()

    def plot_Cl_DeltaP_of_chi_chiprime_2D(self, xmin=None, xmax=None, min_log_range=-20, max_log_range=-9):
        '''
        Plot log |C_l^{\Delta \phi}(\chi,\chi')| against l and  \mathrm{log}_{10}\,\chi
        '''
        if xmin is None:
            xmin = self.field1.chi_mean_fid - 7*self.field1.chi_sigma_fid
        if xmax is None:
            xmax = self.field1.chi_mean_fid + 7*self.field1.chi_sigma_fid

        chi_idx = np.where(np.log10(self.grid.chi_array) > 3.2)[0][0]
        X, Y = np.meshgrid(np.arange(len(self.grid.chi_array)), np.arange(self.lmax + 1))

        Z_fixed_chi = np.log10(np.abs(self.Cl_deltap_of_chi1_chi2[:, :, chi_idx]))
        # Set the log of 0 to a tiny negative number
        Z_fixed_chi[np.abs(self.Cl_deltap_of_chi1_chi2[:, :, chi_idx]) == 0] = -1e100

        # Smooth array with a Gaussian filter for plotting purposes
        Z_fixed_chi_smoothed = gaussian_filter(Z_fixed_chi, sigma=3)

        contours = np.linspace(min_log_range, max_log_range, 20)
        plt.contourf(X, Y, Z_fixed_chi_smoothed, levels=contours, cmap='inferno', extend='both')

        plt.xlabel(r'$\mathrm{log}_{10}\,\chi$')
        plt.ylabel(r'$l$')

        ax = plt.gca()
        ax.axvline(np.where(self.grid.chi_array > self.field1.chi_mean_fid)[0][0], color='r', ls='--', lw=1,
                   label=r'$r=r(z_{\mathrm{mean}})$')

        ax.set_ylim([0, self.lmax])

        label_locs = np.round(np.linspace(xmin+50, xmax-50, 3, dtype=int), decimals=-1)
        ax.set_xticks(utils.find_closest_indices(self.grid.chi_array, label_locs))
        ax.set_xticklabels(label_locs.astype('str'))

        plt.title(r'$\mathrm{log}_{10} \, |C_l^{\Delta '+self.label_dict[self.field1.__class__.__name__]+'}(\chi,\chi^{*})|$')
        plt.colorbar(location='right')

        ax.axvline(chi_idx, label=r'$\chi^{*}$', ls=':', color='w')
        plt.legend()
        plt.xlim([np.where(self.grid.chi_array > xmin)[0][0], np.where(self.grid.chi_array > xmax)[0][0]])
        plt.show()

    def plot_Cl_DeltaP_vs_chi_chiprime_2D(self, l, color_range=None, xmin=None, xmax=None):
        '''
        Plot C_l^{\Delta \phi}(\chi,\chi') against \chi \chi' a fixed l
        '''
        if xmin is None:
            xmin = self.field1.chi_mean_fid - 5*self.field1.chi_sigma_fid
        if xmax is None:
            xmax = self.field1.chi_mean_fid + 5*self.field1.chi_sigma_fid            
        
        suffix = ''

        X, Y = np.meshgrid(np.arange(len(self.grid.chi_array)), np.arange(len(self.grid.chi_array)))
        Z = self.Cl_deltap_of_chi1_chi2[l, :, :]
        
        if color_range is None:
            color_range = np.max(Z)
        contours = np.linspace(-color_range, color_range, 20)

        plt.contourf(X, Y, Z, cmap='RdBu', levels=contours, extend='neither')

        ax = plt.gca()
        ax.axvline(np.where(self.grid.chi_array > self.field1.chi_mean_fid)[0][0],
                   color='k', ls=':', lw=1, label=r'$\chi=\chi_0$')
        ax.axhline(np.where(self.grid.chi_array > self.field1.chi_mean_fid)[0][0],
                   color='k', ls=':', lw=1)

        plt.xlabel(r'$\chi_1$ [Mpc]')
        plt.ylabel(r'$\chi_2$ [Mpc]')

        label_locs = np.round(np.linspace(xmin+50, xmax-50, 3, dtype=int), decimals=-1)
        ax.set_xticks(utils.find_closest_indices(self.grid.chi_array, label_locs))
        ax.set_yticks(utils.find_closest_indices(self.grid.chi_array, label_locs))

        ax.set_xticklabels(label_locs.astype('str'))
        ax.set_yticklabels(label_locs.astype('str'))
        plt.legend()

        plt.title(r'$ C_\ell^{\Delta '+self.label_dict[self.field1.__class__.__name__]+
                  ' \Delta '+self.label_dict[self.field2.__class__.__name__]+'}(\chi_1, \chi_2)$ for $\ell=$' + str(l))
        cbar = plt.colorbar(location='right')
        cbar_labels = np.linspace(np.min(contours), np.max(contours), 5, dtype=float)
        cbar.set_ticks(cbar_labels)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)

        plt.xlim([np.where(self.grid.chi_array > xmin)[0][0], np.where(self.grid.chi_array > xmax)[0][0]])
        plt.ylim([np.where(self.grid.chi_array > xmin)[0][0], np.where(self.grid.chi_array > xmax)[0][0]])
        plt.show()