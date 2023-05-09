#!/global/homes/a/ab2368/.conda/envs/varying_dndz python
import sys
sys.path.insert(0, '/global/homes/a/ab2368/varying_dndzs/Varying_dndzs/biases_code/')
import numpy as np
import biases

# Load scenario from file
sigma_zshifts = np.array([2, 1, 0.5, 0.1])*0.02703
nsides = np.array([8, 16, 32])

# Hyperparameters for numerical integration
num_processes = 1
miniter = 1000
maxiter = 5000
tol = 1e-11
lprime_max = 'none'

# The ells where we want to evaluate the spectra
ells = np.logspace(np.log10(50), np.log10(1500), 48, dtype=int)

count = 0
for sigma_zshift in sigma_zshifts:
    for nside in nsides:
        print('running with nside={}, sigma_zshift={}'.format(nside, sigma_zshift))
        filename = '/pscratch/sd/a/ab2368/data/redmagic_wn_NOSMOOTHING_sigmazshift{}_nside{}'.format(sigma_zshift, nside)
        loaded_exp = biases.load_from_file(filename)
        loaded_exp.ells = ells
        if count==0:
            # In all these cases, the unbiased PS is the same, so we only calculate it once
            unbiased_clgg = biases.unbiased_term(loaded_exp, loaded_exp.ells, num_processes=num_processes, miniter=miniter, maxiter=maxiter, tol=tol)
        loaded_exp.unbiased_clgg = unbiased_clgg
        print('running with lprime_max={}'.format(lprime_max))
        
        # Compute full mode-coupling bias
        loaded_exp.conv_bias = biases.mode_coupling_bias(loaded_exp, loaded_exp.ells, lprime_max=lprime_max, num_processes=num_processes,
                                                         miniter=miniter, maxiter=maxiter, tol=tol)
        # Compute analytic approximations to mode-coupling bias
        loaded_exp.analytic_mcbias_toy_model = biases.mode_coupling_bias(loaded_exp, loaded_exp.ells, num_processes=num_processes, 
                                                                         miniter=miniter, maxiter=maxiter, tol=tol, mode='analytic_toy_model')
        loaded_exp.analytic_mcbias_via_variance = biases.mode_coupling_bias(loaded_exp, loaded_exp.ells, num_processes=num_processes, 
                                                                            miniter=miniter, maxiter=maxiter, tol=tol, mode='analytic_via_variance')

        # Compute additive bias
        loaded_exp.additive_bias = biases.additive_bias(loaded_exp, loaded_exp.ells, num_processes=num_processes, miniter=miniter, maxiter=maxiter, tol=tol)

        # Save to file, this time including the biases as an attribute!
        loaded_exp.save_properties(filename)
        print('done with {}'.format(filename))
        count += 1
