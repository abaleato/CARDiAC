import numpy as np

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