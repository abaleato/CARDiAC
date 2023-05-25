# CARDiAC

Code for Anisotropic Redshift Distributions in Angular Clustering

## Installation
The code has the following dependencies:
- `numpy`, `scipy`, `matplotlib`
- `astropy`
- `healpy`
- `camb`, `anzu` (though the user could replace them with their own power spectra)
- `numba` for JIT compilation of galaxy lensing kernels, which are slow to compute otherwise

To install it in editable mode:
`python -m pip install -e .`

## Usage
More will come here, but for now see `Tutorial.ipynb`.

## Attribution
If you use the code, please cite [Baleato Lizancos & White 2023](https://arxiv.org/abs/2305.15406).