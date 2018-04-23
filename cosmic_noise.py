import numpy as np
from scipy.special import gamma as Gamma
from scipy.stats import rv_continuous

class Brained(rv_continuous):
    '''
        Magnitude limited redshift distribution from Brainerd et al. 1996

        n(z) \propto z**2. * exp(z/z_0*beta)
    '''
    def _pdf(self, x, z_0, beta):
        return beta* (x**2.) * exp(-beta*x/z_0)/Gamma(3./beta)/z_0**3.

def P_k(z_s=None, pw=None):
    '''
        Return an interpolator for the projected convergence power spectrum
        P_k(l), given an input redshift distribution.

        Parameters
        ----------
            z_s : float
                One of the possible inputs, the redshift of the source plane
                (assumes plane approximation)

            W : function
                Distribution of the ratio of angular diameter distance
                D_ls/D_s as a function of the comoving distance to the lens chi
                (assuming Planck15 cosmology). Used only if z_s is not
                specified.

                For how to compute W(chi) given a redshift distribution
                see Contigiani+ 2018.
    '''
    from astropy.cosmology import Planck15 as cosmo
    import astropy.constants as const
    from scipy.interpolate import interp1d
    import camb

    #Hubble parameter
    H_0 = 67.5*u.km/u.s # Technically also /u.Mpc, but here distances are all in Mpc.
    Omega_m = 0.312
    prefactor = (9*H_0**4.*Omega_m**2./4../const.c**4.).to(1).value

    #Obtain matter power spectrum
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.120)
    pars.InitPower.set_params(ns=0.965)
    #Camb returns the power spectrum (in Mpc-3) as a function of k (in Mpc)
    Pk = camb.get_matter_power_interpolator(pars, zmax=3, hubble_units=False, k_hunit=False)

    #Comoving distances
    z_array = np.linspace(0, 10., 200)
    chi_array = cosmo.comoving_distance(z_array).to('Mpc').value
    z = interp1d(chi_array, z_array)

    if(z_s is not None):
        chi_s = cosmo.comoving_distance(z_s).to('Mpc').value
        W = lambda w: 1.-w/chi_s
    else:
        print "Not implemented!"


    #Projected power spectrum
    ls = np.geomspace(l_min_int, l_max_int, 20)
    Pls = np.zeros(20)
    for j in xrange(len(ls)):
        l = ls[j]
        tempint = lambda w: (1.+z(w))**2. * W(w)**2. * np.exp(Pk(z(w), np.log(l/w)))
        Pls[j] = quad(tempint, 0., chi_s)[0]

    return interp1d(ls, Pls)


def CLSS(self, bin_edges, z_s, l_min_int=20, l_max_int=1e4, h=1.):
    '''
        Computes the cosmic noise covariance matrix for tangential shear.
        Hardcoded cosmology: Planck15

        Parameters
        ----------
            bin_edges : Quantity
                Egdes of the angular bins.

            (?)P_k : function
                Convergence angular momentum P_k(l), By default, a fit to the
                COSMOS data is used.

            l_min_int, l_max_int : float
                Limits for the multipole integrals

            souce_z : bool
                Flag for single plane lens approximation with effective source
                redshift z_s

            h : float
                The value of H_0/(100 Mpc/km/s) assumed for bin_edges.
    '''
    from scipy.special import jn
    from scipy.integrate import quad


    # Define auxiliary function
    def g(l, t1, t2):
        a1 = (1.-2.*np.log(t1))/np.pi/(t2**2.-t1**2.)
        a2 = -2./np.pi/(t2**2.-t1**2.)
        a3 = (2.*np.log(t2)-1.)/np.pi/(t2**2.-t1**2.)
        inttemp = lambda x: x*np.log(x)*jn(0, l*x)
        return a1 * t1*jn(1, l*t1)/l  +  a3 * t2*jn(1, l*t2)/l  +  a2 * quad(inttemp, t1, t2)[0]

    covariance_matrix = np.zeros((nbin, nbin))

    nbin = len(bin_edges)-1
    thetamin = (bin_edges[:-1]*.675/h).to('radian').value
    thetamax = (bin_edges[1:]*.675/h).to('radian').value

    P_k = P_k(z_s)
    for j in xrange(nbin):
        for k in xrange(j+1):
            inttemp = lambda l: l*P_k(l)*g(l, thetamin[j], thetamax[j])*g(l, thetamin[k], thetamax[k])
            covariance_matrix[j, k] = quad(inttemp, l_min_int, l_max_int)[0]*2.np.pi

    for j in xrange(nbin):
        for k in xrange(nbin):
            if(k>j):
                covariance_matrix[i][j, k] = covariance_matrices[i][k, j]

    return covariance_matrix
