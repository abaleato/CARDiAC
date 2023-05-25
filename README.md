# CARDiAC

**C**ode for **A**nisotropic **R**edshift **D**istributions **i**n **A**ngular **C**lustering

CARDiAC is a python code that computes the impact of anisotropic redshift distributions on a wide class of angular
 clustering observables, following [Baleato Lizancos & White 2023](https://arxiv.org/abs/2305.15406).
  
At present, the code supports auto- and cross-correlations of galaxy samples and cosmic shear maps, including galaxy
-galaxy lensing. The anisotropy can be present in the mean redshift and/or width of Gaussian distributions, as
   well as in the fraction of galaxies in each component of multi-modal distributions. Templates of these variations
    can be provided by the user or simulated internally within the code.

## Installation
###### Dependencies:
- `numpy`, `scipy`, `matplotlib`
- `astropy`
- `healpy`
- `camb`, `anzu` (though the user could replace them with their own power spectra)
- `numba` for JIT compilation of galaxy lensing kernels, which are slow to compute otherwise

###### Editable installation in-place:
First, clone the repository:

    git clone https://github.com/abaleato/CARDiAC.git   

Then, run:

    python -m pip install -e .

## Usage
See `Tutorial.ipynb`.

## Attribution
If you use the code, please cite [Baleato Lizancos & White 2023](https://arxiv.org/abs/2305.15406).
