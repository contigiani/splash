from cluster import cluster_sample
from astropy import units as u
import numpy as np

CCCP = cluster_sample('data/CCCP.fits', 'data/source/')
#bin_edges = np.array([10,30,60,90,120,150,180,240,300,360,420,480,540,600,720,840,960,1080,1300, 1600])*u.arcsec
bin_edges = np.geomspace(.1, 7, 32)*u.Mpc
CCCP.stack_ESD(bin_edges)

from matplotlib import pyplot as plt

plt.errorbar(CCCP.stack_rbin.value, (CCCP.stack_ESD/CCCP.stack_n).value, yerr=(CCCP.stack_ESDerr/CCCP.stack_n).value, ecolor='#dd0f0f', color='#dd0f0f', lw=4, zorder=100, fmt='o', mew=0)

ESDs = CCCP.ESDs
ESDs[ESDs<=0] = np.nan

for i in xrange(CCCP.size):
    plt.errorbar(CCCP.stack_rbin.value, (ESDs[i]).value, c='0.5', zorder=1)# yerr=(CCCP.ESDs_err[i]).value)

plt.xlabel(r'$R$ $(h^{-1} Mpc)$')
plt.ylabel(r'$\Delta \Sigma (R)$ ($h^{-1}$ '+str(CCCP.ESDs.unit*u.rad)+')')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.gca().set_xlim([bin_edges[0].value, bin_edges[-1].value])
plt.suptitle('Stacked profile')
plt.savefig('stack.pdf')
