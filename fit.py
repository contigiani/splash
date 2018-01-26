from profile import DK14
from cluster import cluster_sample
from astropy import units as u
from matplotlib import pyplot as plt
import emcee
import numpy as np





chain_name = 'wsplash'
max_n = 10000
n_cores = 55

autocorr = np.empty(max_n)
filename = "output/chains/"+chain_name+'.h5'
CCCP = cluster_sample('data/CCCPhighz.fits', 'data/source/')
bin_edges = np.geomspace(.5, 5., 19)*u.Mpc
CCCP.stack_ESD(bin_edges)


x = CCCP.rbin.value
y = CCCP.ESD.value
yerr = CCCP.ESDerr.value



if(chain_name=='wsplash'):

    def lnlike(params, x, y, yerr):
        model = DK14(x, params, 40, mode=0)
        return -0.5*(np.sum(((y-model)/yerr)**2.))


    def lnpost(params, x, y, yerr):
        rho_s, r_s, alpha, rho_0, r_t, beta, gamma, s_e = params
        if(rho_s > 0 and r_s > 0 and alpha > 0 and r_t>0 and beta > 0 and gamma > 0 and rho_0 > 0 and s_e > 0):
            return lnlike(params, x, y, yerr)
        return -np.inf


    ndim, nwalkers = 8, 100
    pos = [[0.02, 2.3, 0.25, 1., 4., 6.,0.01, 4] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
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

if(chain_name == 'wosplash'):

    def lnlike(params, x, y, yerr):
        model = DK14(x, params, 40, mode=1)
        return -0.5*(np.sum(((y-model)/yerr)**2.))


    def lnpost(params, x, y, yerr):
        rho_s, r_s, alpha, rho_0, s_e = params
        if(rho_s > 0 and r_s > 0 and alpha > 0 and rho_0 > 0 and s_e > 0):
            return lnlike(params, x, y, yerr)
        return -np.inf


    ndim, nwalkers = 5, 100
    pos = [[0.02, 2.3, 0.25 ,0.01, 4] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
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
