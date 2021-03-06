import numpy as np
from scipy.special import gamma as Gamma
from scipy.stats import rv_continuous
from astropy import units as u


'''
    Cosmic noise computations, HARDCODED COSMOLOGY
'''


class Brainerd_gen(rv_continuous):
    '''
        Magnitude limited redshift distribution from Brainerd et al. 1996

        n(z) \propto z**2. * exp(-(z/z_0)**beta)

        From rv_continous every instance of Brained_gen inherits methods like
        .fit() (MLE fitting)
        .rvs() (random sampling)

        Check scipy for more info.
    '''
    def _pdf(self, x, z_0, beta):
        return beta* (x**2.) * np.exp(-(x/z_0)**beta )/Gamma(3./beta)/z_0**3.

# This way the fitting can be done easily via Brainerd.fit()
Brainerd = Brainerd_gen(a=0, b=np.inf)

def P_k_gen(z_s=None, z_list=None, weights=None, p_z=Brainerd(z_0 =0.046, beta=0.55).pdf, l_min=1., l_max=1e4,verbose=False):
    '''
        Return an interpolator for the projected convergence power spectrum
        P_kk(l), given an input redshift distribution. For how this is done
        see Contigiani+ 2018.

        Hardcoded WMAP cosmology.

        The default redshift distribution is a fit to
        COSMOS2015 (Laigle et al. 2016) for CCCP-like data.

        Parameters
        ----------
            z_s : float
                One of the possible inputs, the redshift of the source plane
            z_list : np.array
                One of the possible inputs, representative list of source
                redshifts.
            weights : np.array
                Weights for the list z_list. Defaults to constant weighting
            p_z : function
                One of the possible inputs, full redsfhit PDF
            l_min, l_max : float
                P_k(l) is comuted for l_min < l < l_max
    '''
    from astropy.cosmology import Planck15 as cosmo
    import astropy.constants as const
    from scipy.interpolate import interp1d
    from scipy.misc import derivative
    from scipy.integrate import quad
    import camb

    #Hubble parameter
    H_0 = 70*u.km/u.s # Technically also /u.Mpc, but here distances are all in Mpc.
    Omega_m = 0.3
    prefactor = (9*H_0**4.*Omega_m**2./4./const.c**4.).to(1).value

    #Obtain matter power spectrum
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=70, ombh2=0.024, omch2=0.123)
    pars.InitPower.set_params(ns=0.965)
    pars.NonLinear = camb.model.NonLinear_both
    #Camb returns the power spectrum (in Mpc-3) as a function of k (in Mpc)
    Pk = camb.get_matter_power_interpolator(pars, zmax=10, hubble_units=False, k_hunit=False)

    #Comoving distances
    z_array = np.linspace(0, 10., 2000)
    chi_array = cosmo.comoving_distance(z_array).to('Mpc').value
    z = interp1d(chi_array, z_array)

    if(z_s is not None):
        chi_s = cosmo.comoving_distance(z_s).to('Mpc').value
        W = lambda w: 1.-w/chi_s
    else:
        ws = np.linspace(0, 9500, 100)

        if(z_list is not None):
            if(weights is None):
                weights = np.ones(len(z_list))*1./len(z_list)
            w_list = cosmo.comoving_distance(z_list).to('Mpc').value # integration variable
            Ws = np.array([( (1.-w/w_list[w_list > w])*weights[w_list>w] ).sum() for w in ws])
            # Ws is an integral for w'>w of dw' * p(w') * (1-w/w')

        else:
            dz_dw = interp1d(chi_array[:-1], np.diff(z_array)/np.diff(chi_array))
            Ws = np.zeros(100)
            for i in xrange(100):
                inttemp = lambda x: dz_dw(x)*p_z(z(x)) * (1.-ws[i]/x)
                Ws[i] = quad(inttemp, ws[i], 9000)[0]

        W = interp1d(ws, Ws)
        chi_s = 9000

    #Projected power spectrum
    ls = np.geomspace(l_min, l_max, 30)
    Pls = np.zeros(30)
    for j in xrange(len(ls)):
        l = ls[j]
        tempint = lambda w: (1.+z(w))**2. * W(w)**2. * np.exp(Pk(z(w), np.log(l/w)))[0]
        temp = quad(tempint, 0., chi_s)
        Pls[j] = temp[0]*prefactor

    if(verbose):
        return interp1d(ls, Pls), W
    else:
        return interp1d(ls, Pls)

def CLSS(bin_edges, P_k, l_min_int=20, l_max_int=1e4):
    '''
        Computes the cosmic noise covariance matrix for tangential shear.

        Parameters
        ----------
            bin_edges : Quantity
                Egdes of the angular bins.

            P_k : function
                Convergence angular momentum P_k(l), see P_k_gen for
                an example.

            l_min_int, l_max_int : float
                Limits for the multipole integrals
    '''
    from scipy.special import jn
    from scipy.integrate import quad

    nbin = len(bin_edges) -1

    # Define auxiliary function
    def g(l, t1, t2):
        a1 = (1.-2.*np.log(t1))/np.pi/(t2**2.-t1**2.)
        a2 = -2./np.pi/(t2**2.-t1**2.)
        a3 = (2.*np.log(t2)-1.)/np.pi/(t2**2.-t1**2.)
        inttemp = lambda x: x*np.log(x)*jn(0, l*x)
        return a1 * t1*jn(1, l*t1)/l  +  a3 * t2*jn(1, l*t2)/l  +  a2 * quad(inttemp, t1, t2)[0]

    covariance_matrix = np.zeros((nbin, nbin))

    nbin = len(bin_edges)-1
    thetamin = (bin_edges[:-1]).to('radian').value
    thetamax = (bin_edges[1:]).to('radian').value

    for j in xrange(nbin):
        for k in xrange(j+1):
            inttemp = lambda l: l*P_k(l)*g(l, thetamin[j], thetamax[j])*g(l, thetamin[k], thetamax[k])
            covariance_matrix[j, k] = quad(inttemp, l_min_int, l_max_int)[0]*2.*np.pi

    for j in xrange(nbin):
        for k in xrange(nbin):
            if(k>j):
                covariance_matrix[j, k] = covariance_matrix[k, j]

    return covariance_matrix
