import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

class Template:
    def __init__(self, shifts_template, nside, sigma_of_shift, mean_offset=0.):
        """ Contains information about spatial shifts to the center or width of a dndz """
        self.map = shifts_template + mean_offset
        self.npix = len(shifts_template)
        self.nside = nside
        self.sigma_of_shift = sigma_of_shift

    def show(self, temp_plot_min=None, temp_plot_max=None, title=r'$z_{\mathrm{shift}}$'):
        """ Mollview visualization of the shifts template """
        hp.mollview(self.map, min=temp_plot_min, max=temp_plot_max, cmap='PuOr', title=title)
        plt.show()

class MockTemplate(Template):
    def __init__(self, sigma_of_shift, nside, alpha, lmax_pert, mean_offset=0.):
        """ Generate a pixelized map where each pixel value corresponds to a random shift of the
        either the center or the width of the dndz, which is assumed to be Gaussian. The map has
        a standard deviation set by sigma_of_shift, and some underlying power-law power spectrum
        for the shifts with spectral index alpha, truncated at lmax_pert
            - Inputs:
                * sigma_of_shift = float. Standard deviation of the shifts across the template
                * nside = int. Defines the pixelization
                * alpha = float. Power law index for the perturbation Cls
                * lmax_pert = int. lmax of the Cls
                * mean_offset (optional) = float. Mean value about which template must fluctuate
                    (useful for template of interloper fraction, otherwise set to zero by default)
        """
        self.lmax_pert = lmax_pert
        self.alpha = alpha

        pert_cls = np.arange(lmax_pert, dtype=float)**alpha
        pert_cls[0]=0
        # Convert to a map
        sim_map = hp.synfast(pert_cls, nside)
        # Normalize Cls to give desired variance
        norm_factor_for_alms = np.nan_to_num(sigma_of_shift**2 / np.var(sim_map, ddof=1))**0.5
        shifts_template = norm_factor_for_alms*sim_map
        super().__init__(shifts_template, nside, sigma_of_shift, mean_offset)
        
        
class CustomTemplate(Template):
    def __init__(self, shifts_template):
        """ User provides a template of shifts to either the center or the width of a Gaussian dndz
        - Inputs:
            * shifts_template = 1D np.array of floats. A healpy map of shifts
        """
        npix = len(shifts_template)
        nside = hp.npix2nside(npix)
        sigma_of_shift = np.std(shifts_template, ddof=1)
        super().__init__(shifts_template, nside, sigma_of_shift)
