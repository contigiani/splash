
from splashback.profile import DK14
from splashback.cluster import cluster_sample
from astropy import units as u
import emcee
import numpy as np

print "Modules loaded"


# For physical change cosmic_noise, bin_edges comoving, and r_t prior.

catalog_name = "nomergers"
r_limit=True
max_n = 30000
n_cores = 32

autocorr = np.empty(max_n)
if(catalog_name!='simulation'):
    filename = "output/chains/"+catalog_name+'_'+str(r_limit)+'.h5'
    CCCP = cluster_sample('data/CCCP'+catalog_name+'.fits', 'data/source/')
    bin_edges = np.geomspace(.2, 9, 11)*u.Mpc
    CCCP.stack_ESD(bin_edges, cosmic_noise='output/lss_covariance/', comoving=True, mscaling=False, weighted=True, r_limit=r_limit)

    x = CCCP.rbin.value
    y = CCCP.ESD.value
    Cov = CCCP.Cov.value
    x, y, Cov = x, y/1e15, Cov/1e15/1e15

if(catalog_name == 'simulation'):
    filename = "output/chains/"+catalog_name+'.h5'
    x = np.load('output/profiles/simulation_r.npy')
    y = np.load('output/profiles/simulation.npy')
    data_t = np.load('output/profiles/simulation_err.npy')**2.
    Cov = np.diag(data_t)
    x, y, Cov = x, y/1e15, Cov/1e15/1e15


Covinv = np.linalg.inv(Cov)


def normal(x, c, s):
    return -((x-c)**2.)/2./(s**2.)

def lnlike(params, x, y, Covinv):
    model = DK14(x, params, 40.)
    return (-0.5*np.matrix(y-model)*Covinv*np.matrix(y-model).T).sum()


def lnpost(params, x, y, Covinv):
    rho_s, r_s, logalpha, r_t, logbeta, loggamma, rho_0, s_e = params
    params = np.array([rho_s, r_s, logalpha, r_t, logbeta, loggamma, rho_0, s_e])
    if( (0.<rho_s<10.) and (0.<rho_0 < 10.) and (0.1 <= r_s <= 10.)and (0. <= r_t <= 20.) and (1.<s_e<10.)):
        return lnlike(params, x, y, Covinv) + \
                normal(logalpha, np.log10(0.20), 0.1) +\
                normal(logbeta, np.log10(6.), 0.2) +\
                normal(loggamma, np.log10(4.), 0.2) +\
                normal(r_t, 4., 2.)
    return -np.inf


print "initialize chain"
ndim, nwalkers = 8, 31
p0 = np.array([ 0.5,  0.3929062,  -0.27996376,  4.,  0.59970087,  0.79531735, 0.00579822,  1.1855132])
pos = [p0 + p0*np.random.randn(ndim)*1e-4 for i in range(nwalkers)]
backend = emcee.backends.HDFBackend(filename)
first_time = False


try:
    backend.get_chain()
except AttributeError:
    first_time = True
    print "first time"

from multiprocessing import Pool


print "start sampling"
pool = Pool(n_cores)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(x, y, Covinv), backend=backend, pool=pool)
if(first_time):
    sampler.run_mcmc(pos, nsteps=max_n)
else:
    sampler.run_mcmc(pos0=None, nsteps=max_n)
