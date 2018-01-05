from cluster import cluster_sample
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt

CCCP = cluster_sample('data/CCCP.fits', 'data/source/')
bin_edges = np.geomspace(.5, 7, 32)*u.Mpc
CCCP.stack_ESD(bin_edges)

plt.errorbar(CCCP.stack_rbin.value, (CCCP.stack_ESD).value, yerr=(CCCP.stack_ESDerr).value, ecolor='#dd0f0f', color='#dd0f0f', lw=4, zorder=100, fmt='o', mew=0)

ESDs = CCCP.ESDs
ESDs[ESDs<=0] = np.nan

for i in xrange(CCCP.size):
    plt.errorbar(CCCP.stack_rbin.value, CCCP.stack_rbin.value*(ESDs[i]).value, c='0.5', zorder=1)# yerr=(CCCP.ESDs_err[i]).value)

plt.xlabel(r'$R$ $(h^{-1} Mpc)$')
plt.ylabel(r'$\Delta \Sigma (R)$ ($h^{-1}$ '+str(CCCP.ESDs.unit/u.rad)+')')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.gca().set_xlim([bin_edges[0].value, bin_edges[-1].value])
plt.suptitle('Stacked profile')
plt.savefig('output/stack.pdf')
plt.show()

for i in xrange(CCCP.size):
    plt.clf()
    plt.errorbar(CCCP.stack_rbin.value, (CCCP.ESDs[i]).value,yerr=(CCCP.ESDs_err[i]).value, c='0.5', zorder=100)
    plt.xlabel(r'$R$ $(h^{-1} Mpc)$')
    plt.ylabel(r'Rescaled $\Delta \Sigma (R)$ ($h^{-1}$ '+str(CCCP.ESDs.unit/u.rad)+')')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.gca().set_xlim([bin_edges[0].value, bin_edges[-1].value])
    plt.suptitle(CCCP[i].name+' profile')
    plt.savefig('output/individual/'+CCCP[i].name+'.pdf')
