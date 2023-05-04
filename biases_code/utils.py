import numpy as np

def find_closest_indices(chi_array, ref_array):
    overlapping = np.logical_and(ref_array > np.min(chi_array), ref_array < np.max(chi_array))
    out = np.zeros(np.sum(overlapping))
    for i, a in enumerate(ref_array[overlapping]):
        out[i] = np.abs(chi_array - a).argmin()
    return out.astype(int)