from matplotlib import pyplot as plt
from astropy.table import Table
import numpy as np
from astropy import units as u


from astropy.table import Table
import numpy as np
from astropy import units as u

data = Table.read('data/original/clusters.dat', format='ascii')
data_planck = Table.read('data/original/m500_mega_planck.dat', format='ascii')

z, xcen, ycen, mmin, mmax, da, beta_avg, r_500,  m_500, n_0, r_core, r_max = [np.zeros(len(data)) for i in xrange(12)]
column_list = ['name', 'z', 'xcen', 'ycen', 'mmin', 'mmax', 'da', 'beta_avg', 'r_500',  'm_500', 'n_0', 'r_core', 'r_max']
source_column_list = ['x', 'y', 'm', 'e1', 'e2', 'de', 'pg', 'mu', 'delmag']
meta_var = {'details' : 'CCCP sample - Hoekstra+ 2015, Planck r_500, m_500'}
name = [None]*len(data)
da, r_500, m_500, n_0, r_core, r_max = da*u.Gpc/u.rad, r_500*u.Mpc, m_500*u.Msun, n_0*u.arcsec, r_core*u.Mpc, r_max*u.Mpc

i=0
for cluster_name in data['name_cl']:
    print cluster_name
    name[i] = cluster_name
    z[i] = data['z_cl'][data['name_cl']==cluster_name]
    xcen[i] = data['xcen'][data['name_cl']==cluster_name]
    ycen[i] = data['ycen'][data['name_cl']==cluster_name]
    mmin[i] = data['mmin'][data['name_cl']==cluster_name]
    mmax[i] = data['mmax'][data['name_cl']==cluster_name]
    da[i] = data['da'][data['name_cl']==cluster_name]*u.Gpc/u.rad
    beta_avg[i] = data['beta_avg'][data['name_cl']==cluster_name]
    r_500[i] = data_planck['r_d'][data_planck['name_clus']==cluster_name]*u.Mpc
    m_500[i] = data_planck['m_d'][data_planck['name_clus']==cluster_name]*u.Msun

    data_contam = np.loadtxt('data/original/CONTAM_PAR/'+cluster_name+'.par')
    n_0[i] = data_contam[6,4]/data_contam[6, 8]*u.arcsec
    r_core[i] = data_contam[6,6]*u.Mpc*0.7
    r_max[i] = 4 *0.7* u.Mpc
    i+=1

    # LOAD SOURCE CATALOG
    data_source = np.loadtxt('data/original/source/mos_'+cluster_name+'.cat')

    source_meta_var = {'NAME' : cluster_name, 'SAMPLE' : 'CCCP', 'PIXSIZE' : 0.186, 'PIXUNIT' : 'arcsec'}
    Table_source = Table([data_source[:, 0], data_source[:, 1], data_source[:,2], data_source[:, 3], data_source[:,4], data_source[:, 5], data_source[:,6], data_source[:, 7], data_source[:,11]], names=source_column_list, meta=source_meta_var)
    Table_source.write('data/source/'+cluster_name+'.fits', overwrite=True)

#SAVE SAMPLE FITS TABLE
cluster_table = Table([name, z, xcen, ycen, mmin, mmax, da, beta_avg, r_500,  m_500, n_0, r_core, r_max], names=column_list, meta=meta_var)
cluster_table.write('data/CCCP.fits', overwrite=True)
