from cluster import cluster_sample
from astropy import units as u
import numpy as np

CCCP = cluster_sample('data/CCCP.fits', 'data/source/')
bin_edges = np.array([10,30,60,90,120,150,180,240,300,360,420,480,540,600,720,840,960,1080])*u.arcsec
CCCP.stack_ESD(bin_edges)

from matplotlib import pyplot as plt
for i in xrange(CCCP.size):
    data = np.loadtxt('data/original/GTPROF/gtprof_'+CCCP[i].name+'.dat')

    plt.clf()
    plt.errorbar(CCCP[i].rbin.value, CCCP[i].gtbin, yerr=CCCP[i].dgtbin, label='Omar')
    plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 3], label='Henk')
    plt.legend()
    plt.suptitle(CCCP[i].name)
    plt.savefig('output/comparison/'+CCCP.clusters[i].name+'.pdf')
