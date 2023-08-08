import numpy as np
import camb
from camb import model
from astropy.cosmology import Planck18
from scipy.interpolate import interp1d
try:
    # A lot of this is copied from Nick Kokron's anzu repository
    import pyccl as ccl
    from anzu.emu_funcs import LPTEmulator
    from velocileptors.LPT.cleft_fftw import CLEFT
    from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
    def compute_velocileptors_spectra(cosmovec, snapscale, use_physical_densities=True,
                                      use_sigma_8=True, kecleft=True, cleftobj=None):
        '''
        Returns a spline object which computes the cleft component spectra. Computed either in
        "full" CLEFT or in "k-expanded" CLEFT which allows for faster redshift dependence.
        Args:
            cosmovec : array-like
                Vector containing cosmology in the order (ombh2, omch2, w0, ns, sigma8, H0, Neff).
                If self.use_sigma_8 != True, then ln(A_s/10^{-10}) should be provided instead of sigma8.
            snapscale : float
                scale factor
            kecleft: bool
                Bool to check if the calculation is being made with
        Returns:
            cleft_aem : InterpolatedUnivariateSpline
                Spline that computes basis spectra as a function of k
        '''

        if use_physical_densities:
            if use_sigma_8:
                cosmo = ccl.Cosmology(Omega_b=cosmovec[0] / (cosmovec[5] / 100) ** 2,
                                      Omega_c=cosmovec[1] /
                                              (cosmovec[5] / 100) ** 2,
                                      h=cosmovec[5] / 100, n_s=cosmovec[3],
                                      w0=cosmovec[2], Neff=cosmovec[6],
                                      sigma8=cosmovec[4])
            else:
                cosmo = ccl.Cosmology(Omega_b=cosmovec[0] / (cosmovec[5] / 100) ** 2,
                                      Omega_c=cosmovec[1] /
                                              (cosmovec[5] / 100) ** 2,
                                      h=cosmovec[5] / 100, n_s=cosmovec[3],
                                      w0=cosmovec[2], Neff=cosmovec[6],
                                      A_s=np.exp(cosmovec[4]) * 1e-10)
        else:
            if use_sigma_8:
                cosmo = ccl.Cosmology(Omega_b=cosmovec[0],
                                      Omega_c=cosmovec[1] - cosmovec[0],
                                      h=cosmovec[5] / 100, n_s=cosmovec[3],
                                      w0=cosmovec[2], Neff=cosmovec[6],
                                      sigma8=cosmovec[4])
            else:
                cosmo = ccl.Cosmology(Omega_b=cosmovec[0],
                                      Omega_c=cosmovec[1] - cosmovec[0],
                                      h=cosmovec[5] / 100, n_s=cosmovec[3],
                                      w0=cosmovec[2], Neff=cosmovec[6],
                                      A_s=np.exp(cosmovec[4]) * 1e-10)

        k = np.logspace(-3, 1, 1000)

        if kecleft:
            # If using kecleft, check that we're only varying the redshift

            if cleftobj is None:
                # Do the full calculation again, as the cosmology changed.
                pk = ccl.linear_matter_power(
                    cosmo, k * cosmo['h'], 1) * (cosmo['h']) ** 3

                # Function to obtain the no-wiggle spectrum.
                # Not implemented yet, maybe Wallisch maybe B-Splines?
                # pnw = p_nwify(pk)
                # For now just use Stephen's standard savgol implementation.
                cleftobj = RKECLEFT(k, pk)

            # Adjust growth factors
            D = ccl.background.growth_factor(cosmo, snapscale)
            cleftobj.make_ptable(D=D, kmin=k[0], kmax=k[-1], nk=1000)
            cleftpk = cleftobj.pktable.T

        else:
            # Using "full" CLEFT, have to always do calculation from scratch
            pk = ccl.linear_matter_power(
                cosmo, k * cosmo['h'], snapscale) * (cosmo['h']) ** 3
            cleftobj = CLEFT(k, pk, N=2700, jn=10, cutoff=1)
            cleftobj.make_ptable()

            cleftpk = cleftobj.pktable.T

            # Different cutoff for other spectra, because otherwise different
            # large scale asymptote

            cleftobj = CLEFT(k, pk, N=2700, jn=5, cutoff=10)
            cleftobj.make_ptable()

        cleftpk[3:, :] = cleftobj.pktable.T[3:, :]
        cleftpk[2, :] /= 2
        cleftpk[6, :] /= 0.25
        cleftpk[7, :] /= 2
        cleftpk[8, :] /= 2

        cleftspline = interp1d(cleftpk[0], cleftpk, fill_value='extrapolate')

        return cleftspline, cleftobj

    def get_galaxy_ps_anzu(bvec, k, zs_sampled, halomatter=False):
        '''
        Calculate the galaxy power spectrum in the Planck 18 cosmology
        - Inputs:
            * bvec = list containing [b1,    b2,    bs2,   bnabla2, SN] to be fed to Anzu to obtain Pgg
            * z_mean = float. Central redshift of the fiducial dndz
            * k = np array of floats. k at which to evaluate Pkgg.
            * zs_sampled = redshifts at which to evaluate the Anzu prediction
            * halomatter (optional) = Bool. If False, get gg spectrum. If False, get galaxy-matter cross spectrum
        '''
        emu = LPTEmulator()
        h = Planck18.H0.value / 100.

        for i, z in enumerate(zs_sampled):
            a = 1 / (1 + z)
            if i == 0:
                cosmo_vec = np.atleast_2d([Planck18.Ob0 * h ** 2, Planck18.Odm0 * h ** 2, -1, 0.966, 0.812,
                                           Planck18.H0.value, 3.046, a])  # Values from Planck 2018
            else:
                cosmo_vec = np.vstack([np.atleast_2d([Planck18.Ob0 * h ** 2, Planck18.Odm0 * h ** 2, -1, 0.966, 0.812,
                                                      Planck18.H0.value, 3.046, a]), cosmo_vec])

        lpt_spec = np.zeros((len(cosmo_vec), 10, 700))

        # Evaluate predictions at the relevant redshifts
        for i, cv in enumerate(cosmo_vec):
            lpt_interp, cleftobk = compute_velocileptors_spectra(cv, cv[-1],
                                                                 use_physical_densities=emu.use_physical_densities,
                                                                 use_sigma_8=emu.use_sigma_8, kecleft=False)
            lpt_spec[i, ...] = lpt_interp(emu.k)[1:11, :]
        emu_spec = emu.predict(k, cosmo_vec, spec_lpt=lpt_spec)
        Pk = np.zeros((len(k), len(cosmo_vec[:, -1])))
        if halomatter:
            min_idx = len(k)
        else:
            min_idx = 0
        for i, z in enumerate(cosmo_vec[:, -1]):
            Pk[:, i] = emu.basis_to_full(k, bvec, emu_spec[i, :, :], halomatter=halomatter)[min_idx:]
        return Pk
except ImportError:
    print('Anzu/velocileptors/ccl not installed. Proceeding just with CAMB matter PS x linear galaxy bias')

def get_galaxy_ps(g_bias, zs_sampled, g2_bias=None, gbias_mode='linear'):
    '''
    Calculate the galaxy power spectrum
    - Inputs:
        * g_bias = galaxy bias. if gbias_mode=='anzu', a list containing Lagrangian bias [b1,    b2,    bs2,   bnabla2, SN],
                              if gbias_mode=='linear', a float with linear bias value at center of dndz
        * z_mean = float. Central redshift of the fiducial dndz
        * zs_sampled = redshifts at which to evaluate the prediction
        * g2_bias (optional) = Like g_bias, but for the second galaxy sample in spectrum. If None, get galaxy-matter cross-spectrum.
        * gbias_mode (optional) = 'linear' or 'anzu'. Galaxy bias prescription
    '''
    if g2_bias is None:
        halomatter = True
    # ToDo: Choose k's more systematically
    k = np.logspace(-3, 0, 200)
    if gbias_mode=='anzu':
        try:
            # TODO: implement different galaxy bias for two samples in anzu galaxy cross-spectrum
            Pk = get_galaxy_ps_anzu(g_bias, k, zs_sampled, halomatter=halomatter)
            return k, Pk
        except:
            print('Anzu/velocileptors not installed. Proceeding just with CAMB matter PS x linear bias')
    k, pk_nonlin = get_matter_ps(zs_sampled)
    Pk = np.swapaxes(pk_nonlin, 0, 1)
    try:
        if g2_bias is None:
            # Halo-matter cross-spectrum using linear galaxy bias
            return k, Pk * g_bias
        else:
            # Galaxy auto-spectrum using linear galaxy bias
            return k, Pk * g_bias * g2_bias
    except:
        print('Galaxy bias must be a linear (i.e. a single number) when not using anzu')

def get_matter_ps(redshifts):
    #Now get matter power spectra and sigma8 at redshifts between 0 and sufficiently behind the perturbed sources
    pars = camb.CAMBparams()
    h = Planck18.H0.value/100.
    pars.set_cosmology(H0=Planck18.H0.value, ombh2=Planck18.Ob0 * h**2, omch2=Planck18.Odm0 * h**2)
    pars.InitPower.set_params(ns=0.966)
    #Note non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=redshifts, kmax=2.0)

    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1e2, npoints = 500)
    s8 = np.array(results.get_sigma8())

    #Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1e2, npoints = 500)

    # Remove factors of h
    k_nonlin = kh_nonlin * h
    pk_nonlin *= h**3
    return k_nonlin, pk_nonlin
