#!/global/homes/a/ab2368/.conda/envs/varying_dndz python
import sys
sys.path.insert(0, '/global/homes/a/ab2368/varying_dndzs/Varying_dndzs/code/')
import numpy as np
import biases

# Load scenario from file
sigma_zshift = 0.02703
nside = 8
filename = '/pscratch/sd/a/ab2368/data/redmagic_wn_sigmazshift{}_nside'.format(sigma_zshift, nside)
loaded_exp = biases.load_from_file(filename)

# Hyperparameters for numerical integration
num_processes = 16
miniter = 1000
maxiter = 5000
tol = 1e-11

# The ells where we want to evaluate the spectra
loaded_exp.ells = np.logspace(0, np.log10(200), 16, dtype=int)

if hasattr(loaded_exp, 'additive_bias'):
    print("It looks like you've already computed the bias for this scenario. Please delete it from file you really want to re-rerun things.")
else:
    loaded_exp.additive_bias = biases.additive_bias(loaded_exp, loaded_exp.ells, num_processes=num_processes, miniter=miniter, maxiter=maxiter, tol=tol)
    #loaded_exp.unbiased_clgg = biases.unbiased_term(loaded_exp, loaded_exp.ells, num_processes=num_processes, miniter=miniter, maxiter=maxiter, tol=tol)

# Save to file, this time including the biases as an attribute!
loaded_exp.save_properties(filename)
