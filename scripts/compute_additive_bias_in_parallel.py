#!/global/homes/a/ab2368/.conda/envs/varying_dndz python
import sys
sys.path.insert(0, '/global/homes/a/ab2368/varying_dndzs/Varying_dndzs/biases_code/')
import numpy as np
import biases

# Load scenario from file
sigma_zshifts = np.array([0.1])*0.02703 #np.array([2, 1, 0.5, 0.1])*0.02703 #np.array([0.02703])
nsides = np.array([32])#np.array([8, 16, 32])

# Hyperparameters for numerical integration
num_processes = 1
miniter = 1000
maxiter = 5000
tol = 1e-11

for sigma_zshift in sigma_zshifts:
    for nside in nsides:
        filename = '/pscratch/sd/a/ab2368/data/redmagic_wn_NOSMOOTHING_sigmazshift{}_nside'.format(sigma_zshift, nside)
        loaded_exp = biases.load_from_file(filename)

        # The ells where we want to evaluate the spectra
        loaded_exp.additive_ells = np.logspace(0, np.log10(200), 16, dtype=int)

        if hasattr(loaded_exp, 'additive_bias'):
            print("It looks like you've already computed the bias for this scenario. Please delete it from file you really want to re-rerun things.")
        else:
            loaded_exp.additive_bias = biases.additive_bias(loaded_exp, additive_ells.ells, num_processes=num_processes, miniter=miniter, maxiter=maxiter, tol=tol)

        # Save to file, this time including the biases as an attribute!
        loaded_exp.save_properties(filename)
