
import numpy as np
import biases

# Load scenario from file
sigma_zshift = 0.02703
filename = '../data/redmagic_wn_sigmazshift{}'.format(sigma_zshift)
loaded_exp = biases.load_from_file(filename)

# The ells where we want to evaluate the spectra
loaded_exp.ells = np.linspace(10, 1000, 15, dtype=int)

if hasattr(loaded_exp, 'conv_bias'):
    print("It looks like you've already computed the biases for this scenario. Please delete them from file you really want to re-rerun things.")
else:
    loaded_exp.conv_bias = biases.mode_coupling_bias(redmagic_wn, loaded_exp.ells, parallelize=True)
    loaded_exp.additive_bias = biases.additive_bias(redmagic_wn, loaded_exp.ells, parallelize=True)
    loaded_exp.unbiased_clgg = biases.unbiased_term(redmagic_wn, loaded_exp.ells, parallelize=True)

# Save to file, this time including the biases as an attribute!
loaded_exp.save_properties(filename)
