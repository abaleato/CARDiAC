import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

class Field:
    def __init__(self, delta_p_maps, kernel):
        """ Contains information about spatial shifts to the center or width of a dndz """
        
        # Take the spherical harmonic transform of each r slice. Conveniently, we can take complex SHT so array sizes reduce by x2
        delta_p_lm_of_chi = np.zeros((hp.Alm.getsize(3 * template.nside - 1), n_samples_of_chi), dtype=complex)
        for i in range(n_samples_of_chi):
            delta_p_lm_of_chi[:, i] = hp.map2alm(delta_p_maps[:, i])
        


class gal_delta(Field):
    def __init__(self, Template):
        """ Generate a pixelized map where each pixel value corresponds to a random shift of the
            - Inputs:
                * sigma_of_shift = float. Standard deviation of the shifts across the template

        """
        kernel = 

        super().__init__()
        
        
class gal_shear(Field):
    def __init__(self, Template):
        """ User provides a template of shifts to either the center or the width of a Gaussian dndz
        - Inputs:
            * shifts_template = 1D np.array of floats. A healpy map of shifts
        """
        # If doing cosmic shear, calculate the galaxy lensing kernel
            # Dummy run to pre-compile with numba
            utils.lens_efficiency_kernel(self.chi_array, self.chi_max_int, np.zeros_like(self.chi_array))

            # Now in earnest
            self.g_fid = 3 / 2. * Planck18.Om0 * Planck18.H0.value ** 2 / c.value ** 2 * self.chi_array * (
                        1 + self.z_array[0, :]) * utils.lens_efficiency_kernel(self.chi_array, self.chi_max_int,
                                                                               self.phi_fid_array)
            self.g_pert = 3 / 2. * Planck18.Om0 * Planck18.H0.value ** 2 / c.value ** 2 * self.chi_array * (
                        1 + self.z_array[0, :]) * utils.lens_efficiency_kernel(self.chi_array, self.chi_max_int,
                                                                               self.phi_perturbed_array)
        kernel = interp1d(self.chi_array, self.g_fid)

        super().__init__()
