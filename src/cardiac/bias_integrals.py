import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from py3nj import wigner3j
from scipy.integrate import quadrature
import multiprocessing
from functools import partial

#
# First, the unbiased/isotropic case
#

def unbiased_term(exp, ells, num_processes=1, miniter=1000, maxiter=2000, tol=1e-12):
    """ Calculate the unbiased contribution to the angular clustering power spectrum in the Limber approximation
        - Inputs:
            * exp = an instance of the experiment class
            * ells = np array of ints. The ells at which to evaluate this bias
            * num_processes (optional) = int. Number of processes to use. If >1, use multiprocessing
            * miniter (optional) = int. Minimum number of iterations for quadrature.
            * maxiter (optional) = int. Maximum number of iterations for quadrature.
            * tol (optional) = int. Error tolerance before breaking numerical integration.
    """
    if num_processes > 1:
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
    """ Calculate the unbiased contribution to the angular clustering power spectrum in the Limber approximation
        at a specific l.
        - Inputs:
            * exp = an instance of the experiment class
            * miniter (optional) = int. Minimum number of iterations for quadrature.
            * maxiter (optional) = int. Maximum number of iterations for quadrature.
            * tol (optional) = int. Error tolerance before breaking numerical integration.
            * l = int. The multipole of \Delta C_l
    """
    print('Working on l={}'.format(l))
    Pk_interp_1D = interp1d(exp.grid.chi_array, np.diagonal(np.flipud(exp.Pk_interp(exp.grid.chi_array,
                                                                               (l + 0.5) / exp.grid.chi_array))))
    result, error = quadrature(integrand_unbiased_auto_term, exp.grid.chi_min_int, exp.grid.chi_max_int,
                               args=(exp.kernel, Pk_interp_1D), miniter=miniter, maxiter=maxiter, tol=tol)
    return result

def integrand_unbiased_auto_term(chi, kernel, Pk_interp_1D):
    """
    Integrand for the unbiased Cl auto spectrum in the Limber approximation
    """
    return Pk_interp_1D(chi) * kernel(chi) / chi** 2


#
# On to the additive bias to galaxy clustering
#

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
    if num_processes > 1:
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
    """ Calculate the additive bias to the galaxy clustering power spectrum
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
        Clchi1chi2_interp = interpolate.RegularGridInterpolator((exp.grid.chi_array, exp.grid.chi_array),
                                                                exp.Cl_deltap_of_chi1_chi2[l, :, :],
                                                                method='linear', bounds_error=True, fill_value=0)

        result, error = quadrature(integrand_additive_term, exp.grid.chi_min_int,
                                   exp.grid.chi_max_int, args=(Clchi1chi2_interp, exp.grid.chi_min_int, exp.grid.chi_max_int),
                                   miniter=miniter, maxiter=maxiter, tol=tol)
    except IndexError as e:
        print(f"{e}")
        print('Setting the result to 0 because ClDeltaphi ought to be 0 at this l')
        result = 0
    return result

def integrand_additive_term(chi1, Clchi1chi2_interp, chi_min_int, chi_max_int):
    """
    Integrand for the additive bias term, in the Limber approximation
    """
    outer_integral, error = quadrature(integrand_nested_additive_term, chi_min_int,
                                       chi_max_int, args=(chi1, Clchi1chi2_interp), miniter=3,
                                       maxiter=5, tol=1e-20)
    return outer_integral

def integrand_nested_additive_term(chi2, chi1, Clchi1chi2_interp):
    chi1 = chi1[0]
    chi2 = chi2[0]
    return Clchi1chi2_interp(np.array((chi1, chi2)))[0]

#
# And, finally, the mode-coupling term
#

def mode_coupling_bias(exp, ells, lprime_max='none', num_processes=1, miniter=1000, maxiter=2000, tol=1e-12,
                       mode='full'):
    """ Calculate the mode-coupling bias to the angular clustering power spectrum in the Limber approximation
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
    if mode == 'full':
        # Do the full calculation, using no analytic approximations
        integral_at_l = mode_coupling_bias_at_l
        if lprime_max == 'none':
            lprime_max = exp.Cl_deltap_of_chi1_chi2.shape[0] - 1
    elif mode == 'analytic_via_variance':
        # Use the analytic approximation where the Limber kernel is \mathrm{Var}[\phi](\chi)
        exp.mc_kernel = exp.analytic_proj_kernel
        integral_at_l = analytic_mode_coupling_bias_at_l
    elif mode == 'analytic_toy_model':
        # Use the analytic approximation where the Limber kernel is [f(\chi)\bar{\phi}(\chi)]^2
        exp.mc_kernel = exp.analytic_kernel_toy_model
        integral_at_l = analytic_mode_coupling_bias_at_l
    print('Using mode {}'.format(mode))

    if num_processes > 1:
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
    """ Calculate the mode-coupling bias to the angular clustering power spectrum in the Limber approximation,
        at a specific l.
        - Inputs:
            * exp = an instance of the experiment class
            * lprime_max = int. Value of l above which we ignore the anisotropy in C_l^{\Delta \phi}
            * miniter (optional) = int. Minimum number of iterations for quadrature.
            * maxiter (optional) = int. Maximum number of iterations for quadrature.
            * tol (optional) = int. Error tolerance before breaking numerical integration.
            * l = int. The multipole of \Delta C_l
    """
    #TODO: Implement beyond-Limber
    print('Working on l={}'.format(l))
    integrand = np.zeros_like(exp.grid.chi_array)
    cldp_interp = exp.cldp_interp(exp.grid.chi_array)
    for L in range(l + lprime_max + 1):
        abslmL = np.abs(l - L)
        lpL = l + L
        if abslmL <= lprime_max:
            # Only loop if you will probe scales below cut
            Pk_interp = np.diagonal(np.flipud(exp.Pk_interp(exp.grid.chi_array, (L + 0.5) / exp.grid.chi_array)))
            for lprime in np.arange(abslmL, min(lprime_max, lpL) + 1, 1):
                if (l + lprime + L) % 2 == 0:
                    w3 = wigner3j(2 * l, 2 * L, 2 * lprime, 0, 0, 0)
                    prefactor = w3 ** 2 * (2 * lprime + 1) * (2 * L + 1) / (4 * np.pi)
                    integrand += prefactor / exp.grid.chi_array ** 2 * Pk_interp * cldp_interp[lprime]
    f = interp1d(exp.grid.chi_array, integrand)
    result, error = quadrature(f, exp.grid.chi_min_int, exp.grid.chi_max_int, miniter=miniter, maxiter=maxiter, tol=tol)
    return result

def analytic_mode_coupling_bias_at_l(exp, dummy, miniter, maxiter, tol, l):
    """ Calculate the mode-coupling bias to the angular clustering power spectrum in the Limber approximation,
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
    Pk_interp_1D = interp1d(exp.grid.chi_array, np.diagonal(np.flipud(exp.Pk_interp(exp.grid.chi_array,
                                                                               (l + 0.5) / exp.grid.chi_array))))
    result, error = quadrature(limber_integral, exp.grid.chi_min_int,
                               exp.grid.chi_max_int, args=(Pk_interp_1D, exp.mc_kernel),
                               miniter=miniter, maxiter=maxiter, tol=tol)
    return result

def limber_integral(chi, Pk_interp_1D, kernel):
    return kernel(chi) * Pk_interp_1D(chi)






