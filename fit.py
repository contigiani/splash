from profile import DK14
from cluster import cluster_sample
from astropy import units as u
from matplotlib import pyplot as plt
import emcee
import numpy as np





chain_name = 'wsplash'
max_n = 10000
n_cores = 4

autocorr = np.empty(max_n)
filename = "output/chains/"+chain_name+'.h5'
CCCP = cluster_sample('data/CCCP49.fits', 'data/source/')
bin_edges = np.geomspace(.3, 7., 14)*u.Mpc
CCCP.stack_ESD(bin_edges)


x = CCCP.rbin.value
y = CCCP.ESD.value
yerr = CCCP.ESDerr.value

def normal(x, c, s):
    return -((x-c)**2.)/2./(s**2.)


if(chain_name=='wsplash'):

    def lnlike(params, x, y, yerr):
        model = DK14(x, params, 40,0.3, mode=0)
        return -0.5*(np.sum(((y-model)/yerr)**2.))


    def lnpost(params, x, y, yerr):
        mu, rho_s, r_s, logalpha, r_t, logbeta, loggamma, rho_0, s_e = params
        if( (mu>0.) and (0.<rho_s<1.) and (0.<rho_0 < 1.) and (0.1 <= r_s <= 5.) and (0.1<=r_t<=5.) and (1.<s_e<10.)):
            return lnlike(params, x, y, yerr) + \
                    normal(logalpha, np.log10(0.2), 0.6) +\
                    normal(logbeta, np.log10(4), 0.2) +\
                    normal(loggamma, np.log10(6), 0.2)
        return -np.inf


    ndim, nwalkers = 9, 36
    pos = [[0.05, 0.01, 2.3, np.log10(0.25), 1., np.log10(4.), np.log10(6.), 0.01, 4.] + 5e-3*np.random.randn(ndim) for i in range(nwalkers)]
    backend = emcee.backends.HDFBackend(filename)
    first_time = False

    try:
        backend.get_chain()
    except AttributeError:
        first_time = True
        #print 'Initializing new chain'

    from multiprocessing import Pool

    pool = Pool(n_cores)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(x, y, yerr), backend=backend, pool=pool)
    if(first_time):
        sampler.run_mcmc(pos, nsteps=max_n)#, progress=True)
    else:
        sampler.run_mcmc(pos0=None, nsteps=max_n)#, progress=True)
