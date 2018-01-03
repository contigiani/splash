from astropy import units as u
import numpy as np
from astropy.units import Quantity

'''
    Splashback radius

    Assumes all distances to be conmoving, Assumes Hoekstra+ 2015 cosmology with h=1.
'''


class cluster_sample:
    '''
        Load a cluster sample from directory


        Parameters
        ----------
        dirpath : str
            String of the directory where to find the fits tables for the clusters.
        parallel : bool
            If True, perform the loading using n-1 cores, where n=multiprocessing.cpu_count()-1.
            Requires multiprocessing. Default value: True.
    '''
    def __init__(self, dirpath, parallel=True):
        import glob

        filelist = glob.glob(dirpath+'/*.fits')
        self.size = len(filelist)
        self.clusters = []*self.size


        if(parallel):
            from joblib import Parallel, delayed
            import multiprocessing
            from multiprocessing import Pool , cpu_count
            n = cpu_count()-1
            p = Pool(cpu_count()-1)
            self.clusters = p.map(cluster, filelist)
        else:
            for i in xrange(self.size):
                filename = filelist[i]
                self.clusters[i] = cluster(filename)


    def stack_ESD(self, bin_edges=None, idxlist=None):
        '''
            Stack scaled ESDs

            Parameters
            ----------
            bin_edges : Quantity or float
                Edges of the azimuthal bins to average over. Can be either a distance, comoving or
            idxlist : list
                Indices of the array to stack. Defaults to stacking all clusters.
        '''

        if(idxlist == None):
            idxlist = range(self.size)

        bin_edges = Quantity(bin_edges)
        if(bin_edges.unit.is_equivalent('1')):
            [self.clusters[i].compute_shear(bin_edges*self.clusters[i].r_d) for i in xrange(self.size)]
        else:
            [self.clusters[i].compute_shear(bin_edges) for i in xrange(self.size)]

        self.ESDs, self.ESDs_err = np.zeros([self.size, len(bin_edges)-1]), np.zeros([self.size, len(bin_edges)-1])
        for i in xrange(self.size):
            if(i in idxlist):
                self.ESDs[i] = self.clusters[i].gtbin*self.clusters[i].da/self.clusters[i].beta_avg/self.clusters[i].m_d
                self.ESDs_err[i] = self.clusters[i].dgtbin*self.clusters[i].da/self.clusters[i].beta_avg/self.clusters[i].m_d

        rmin = bin_edges[:-1]
        rmax = bin_edges[1:]

        self.stack_rbin = 0.667*(rmax**3-rmin**3)/(rmax**2-rmin**2) # area-weighted average
        self.stack_ESD = self.ESDs.sum(0)
        self.stack_ESDerr = np.sqrt((self.ESDs_err**2.).sum(0))
        self.stack_n = (self.ESDs != 0).sum(0)

class cluster:
    '''
        Load a cluster from file

        Parameters
        ----------
        filepath : str
            Path of the astropy table file

        rbin : ndarray
            Radial bins to compute azimuthally averaged shear components around the center of the cluster
        gtbin : ndarray
            Tangential components computed using rbin
        gxbin : ndarray
            Cross-components computed using rbin
        dgtbin : ndarray
            Error on the tangential component
    '''
    metavar_list = ['name', 'sample', 'pixscale', 'z', 'xcen', 'ycen', 'mmin', 'mmax', 'da', 'beta_avg', 'r_d', 'r_d_high', 'r_d_low', 'm_d', 'm_d_high', 'm_d_low', 'mproj_d','dmproj_d']
    column_list = ['x', 'y', 'm', 'e1', 'e2', 'de', 'pg', 'mu', 'rh', 'nu', 'ncoin', 'delmag']

    def __init__(self, filepath):
        from astropy.table import Table

        data_table = Table.read(filepath)
        data_table.meta =  {k.lower(): v for k, v in data_table.meta.items()}
        self.rbin = None
        self.gtbin = None
        self.gxbin = None
        self.filepath = filepath

        #Load metavariables
        for metaname in data_table.meta:
            try:
                i = self.metavar_list.index(metaname)
                setattr(self, metaname, data_table.meta[metaname])
                i+=1
            except ValueError:
                print('Metavariable not recognized:' + str(metaname))
                i+=1
                continue

        self.pixscale = self.pixscale * u.arcsec
        self.da = self.da * u.Gpc / u.rad
        self.r_d = self.r_d * u.Mpc

        #Load columns
        for colname in data_table.colnames:
            try:
                i = self.column_list.index(colname)
                setattr(self, colname, data_table[colname].quantity)
                i+=1
            except ValueError:
                print('Column not recognized: ' + str(colname))
                i+=1
                continue


    def compute_shear(self, bin_edges=None):
        '''
            Compute tangential and cross component of the shear around the
            cluster center. The results are stored in gtbin and gxbin, the error
            on the tangential shear is stored in dgtbin.

            Does not take into account the contamination from cluster members.

            Parameters
            ----------
            bin_edges : Quantity or float
                 Array or the bin edges. For bins of size N,
                 gtbin and gxbin will have size N-1.
        '''


        idx = (self.m > self.mmin) & (self.m < self.mmax) & (self.pg > 0.1) #& (delmag==0)
        x, y, e1, e2, pg, de, m, mu = self.x[idx], self.y[idx], self.e1[idx], self.e2[idx], self.pg[idx], self.de[idx], self.m[idx], self.mu[idx]

        #Transform x and y in arcsec
        x = self.pixscale*(x - self.xcen)
        y = self.pixscale*(y - self.ycen)
        r = np.sqrt(x**2.+y**2.)
        pg[pg<0.] = 0


        #weights
        w = pg**2./((0.25*pg)**2. + de**2.)
        wg = pg/((0.25*pg)**2. + de**2.)
        et = -((x*x-y*y)*e1 + 2.0*x*y*e2)/r**2.
        ex = ((x*x-y*y)*e2-2.0*x*y*e1)/r**2

        bin_edges = Quantity(bin_edges)

        if(bin_edges.unit.is_equivalent('Mpc')):
            r = r * self.da
        else:
            r = r

        rmin = bin_edges[:-1]
        rmax = bin_edges[1:]

        nbin = len(rmin)
        rbin = 0.667*(rmax**3-rmin**3)/(rmax**2-rmin**2) # area-weighted average
        nbin, gtbin, gxbin, dgtbin, kbin = [np.zeros(nbin) for i in xrange(5)]


        for i in xrange(len(rmin)):
            idx = (r < rmax[i]) & (r > rmin[i])
            nbin[i] = idx.sum()
            gtbin[i] = np.sum(wg[idx]*et[idx])/np.sum(w[idx])
            gxbin[i] = np.sum(wg[idx]*ex[idx])/np.sum(w[idx])
            dgtbin[i] = np.sqrt(1./np.sum(w[idx]))
            kbin[i] = np.sum(mu[idx]*w[idx])/np.sum(w[idx])

        self.rbin = rbin
        self.gtbin = gtbin/kbin
        self.gxbin = gxbin/kbin
        self.dgtbin = dgtbin
        self.nbin = nbin

    def print_info(self):
        print 'Cluster name: '+self.name
        print 'Source catalog: '+self.filepath
        print 'Pixel position of cluster center: '+self.xcen+', '+self.ycen
        print 'Full Metavariable: '+self.meta
