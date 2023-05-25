import numpy as np
import healpy as hp
import numba as nb

def find_closest_indices(chi_array, ref_array):
    overlapping = np.logical_and(ref_array > np.min(chi_array), ref_array < np.max(chi_array))
    out = np.zeros(np.sum(overlapping))
    for i, a in enumerate(ref_array[overlapping]):
        out[i] = np.abs(chi_array - a).argmin()
    return out.astype(int)

def bl(fwhm_arcmin, lmax):
    """ returns the map-level transfer function for a symmetric Gaussian beam.
         * fwhm_arcmin      = beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax             = maximum multipole.
    """
    ls = np.arange(0, lmax + 1)
    return np.exp(-(fwhm_arcmin * np.pi / 180. / 60.) ** 2 / (16. * np.log(2.)) * ls * (ls + 1.))

def generate_template(nside, alpha, target_std, lmax_pert):
    '''
    Generate a pixelized map where each pixel value corresponds to a random shift of the
    center of the dndz. The map has a standard deviation set by target_std, and some underlying
    power-law power spectrum for the shifts
    Inputs:
        * nside = int. Defines the pixelization
        * alpha = float. Power law index for the perturbation Cls
        * target_std = float. Standard deviation of the resulting map
        * lmax_pert = int. lmax of the Cls
    '''
    pert_cls = np.arange(lmax_pert, dtype=float)**alpha
    pert_cls[0]=0

    # Convert to a map
    sim_map = hp.synfast(pert_cls, nside)

    # Normalize Cls to give desired variance
    norm_factor_for_alms = np.nan_to_num(target_std**2 / np.var(sim_map, ddof=1))**0.5
    return norm_factor_for_alms*sim_map

@nb.jit(nopython=True)
def integral_at_chi(chi, chi_max_int, phi_array, chi_s):
    min_idx = np.where(chi_s>=chi)[0][0]

    if phi_array.ndim==2:
        outs_at_chi = np.zeros(phi_array.shape[0], dtype=float)
        for j in range(phi_array.shape[0]):
            # Iterate over healpixels
            integrand = phi_array[j,:] * (chi_s - chi)/chi_s
            outs_at_chi[j] = np.trapz(integrand[min_idx:], chi_s[min_idx:])
    else:
        integrand = phi_array * (chi_s - chi)/chi_s
        outs_at_chi = np.trapz(integrand[min_idx:], chi_s[min_idx:])
    return outs_at_chi

@nb.jit(nopython=True)
def lens_efficiency_kernel(chis, chi_max_int, phi_array):
    output = np.zeros_like(phi_array)
    if phi_array.ndim==1:
        for i, chi in enumerate(chis):
            output[i] = integral_at_chi(chi, chi_max_int, phi_array, chis)
    else:
        for i, chi in enumerate(chis):
            output[:, i] = integral_at_chi(chi, chi_max_int, phi_array, chis)
    return output