from cluster import cluster_sample
from astropy import units as u
import numpy as np

a = cluster_sample('data/source/')

bin_edges = np.geomspace(.1, 7., 10)*u.Mpc #no units => bins as fractions of R_500c
a.stack_ESD(bin_edges)

from matplotlib import pyplot as plt
plt.errorbar(a.stack_rbin.value, a.stack_ESD*1e15/a.stack_n, yerr=a.stack_ESDerr*1e15/a.stack_n, c='k', lw=4, zorder=100)

for i in xrange(a.size):
    if((a.ESDs[i] > 0).all()):
        plt.errorbar(a.stack_rbin.value, a.ESDs[i]*1e15, yerr=a.ESDs_err[i]*1e15, c='0.5', label=a.clusters[i].name, zorder=1)

plt.legend()
plt.xlabel(r'$R$ $(h^{-1} Mpc)$')
plt.ylabel(r'$\Delta \Sigma (R)$ (arbitrary units)')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.show()
