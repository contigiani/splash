from astropy import units as u
import numpy as np
from astropy.units import Quantity

'''
    Splashback radius for KSB source catalogs

    Assumes all distances to be comoving, Hoekstra+ 2015 cosmology (h=1, O_m = 0.3, O_L = 0.7).
'''


class cluster_sample:
    '''
        Load a cluster sample from a Table


        Parameters
        ----------
        filepath : str
            Path where to find the fits table for the cluster sample.
        dirpath : str
            Path where to find the directory with the source catalogs in fits format.
        parallel : bool
            If True, perform the loading using n-1 cores, where n=multiprocessing.cpu_count().
            Requires multiprocessing. Default value: True.

        Methods
        -------
        stack_ESD
            Stacks the Excess surface densities of the individual clusters,

            computes the following properties:
            - self.rbin average position per radial bin
            - self.ESD value of the ESD for the bins, units Msun/Mpc/Mpc/h
            - self.ESD_stat_err statistical error on self.ESD
            - self.Cov covariance matrix including cosmic noise (optional)
            - self.ESD_err (diagonal of Cov)
            - self[i].gt tangential shear for i-th cluster
            - self[i].dgt error in the tangential shear for i-th cluster
    '''

    filepath = None
    size = 0


    def __init__(self, filepath, dirpath, parallel=True):
        from astropy.table import Table

        self.filepath = filepath
        self.dirpath = dirpath
        data = Table.read(filepath)

        self.size = len(data)
        self.clusters = [None]*self.size

        # Load cluster catalog
        varnames = ['r_500', 'm_500', 'beta_avg', 'z', 'da', 'z_s']

        for varname in varnames:
            setattr(self, varname, data[varname].quantity)

        # Load sources
        filelist = [dirpath+data['name'][i]+'.fits' for i in xrange(self.size)]
        for i in xrange(self.size):
            self.clusters[i] = cluster(filelist[i])
            varnames = ['xcen', 'ycen', 'mmin', 'mmax']
            for varname in varnames:
                setattr(self.clusters[i], varname, data[varname][i])

            varnames = ['da', 'n_0', 'r_core', 'r_max', 'r_500', 'z']
            for varname in varnames:
                setattr(self.clusters[i], varname, data[varname].quantity[i])

    def stack_ESD(self, bin_edges=None, raw=False, mscaling=False, contamination=True, comoving=False, cosmic_noise=None, weighted=False):
        '''
            Stack scaled ESDs

            Parameters
            ----------
            bin_edges : Quantity or float
                Edges of the azimuthal bins to average over. Can be either a distance, comoving or
            raw : bool
                Compute the ESD using PSF-uncorrected ellipticities. Defaults to False
            mscaling : bool
                Scales individual ESD by M500. Defaults to False
            contamination : bool
                Compute the cluster member contamination and obscuration corrections for individual clusters. Defaults to True
            comoving : bool
                Stack in comoving coordinates for bin_edges. Defaults to False
            cosmic_noise : function or string
                Compute cosmic noise covariance matrix if a projected
                convergence power spectrum P_k(l) is provided as a function.
                Alternatively, a path to to the cosmic noise tangential shear
                covariance matrices can also be provided.

                See the cosmic_noise module for more info. Notice that this
                works only when stacking in physical units.
        '''
        import astropy.constants as const

        bin_edges = Quantity(bin_edges)
        if(bin_edges.unit.is_equivalent('1')):
            [self.clusters[i].compute_shear(bin_edges*self.r_500[i], raw, contamination, comoving) for i in xrange(self.size)]
        else:
            [self.clusters[i].compute_shear(bin_edges, raw, contamination, comoving) for i in xrange(self.size)]

        if(mscaling):
            self.ESDs, self.ESDs_err = np.zeros([self.size, len(bin_edges)-1])/u.Mpc/u.Mpc, np.zeros([self.size, len(bin_edges)-1])/u.Mpc/u.Mpc
        else:
            self.ESDs, self.ESDs_err = np.zeros([self.size, len(bin_edges)-1])*u.Msun/u.Mpc/u.Mpc, np.zeros([self.size, len(bin_edges)-1])*u.Msun/u.Mpc/u.Mpc

        for i in xrange(self.size):
            if(mscaling):
                self.ESDs[i] = self.clusters[i].gtbin/self.da[i]/self.beta_avg[i]/self.m_500[i]*(const.c**2.)/4./np.pi/const.G/u.rad
                self.ESDs_err[i] = self.clusters[i].dgtbin/self.da[i]/self.beta_avg[i]/self.m_500[i]*(const.c**2.)/4./np.pi/const.G/u.rad
            else:
                self.ESDs[i] = self.clusters[i].gtbin/self.da[i]/self.beta_avg[i]*(const.c**2.)/4./np.pi/const.G/u.rad
                self.ESDs_err[i] = self.clusters[i].dgtbin/self.da[i]/self.beta_avg[i]*(const.c**2.)/4./np.pi/const.G/u.rad

        rmin = bin_edges[:-1]
        rmax = bin_edges[1:]

        self.ESDs[~np.isfinite(self.ESDs)] = 0
        self.ESDs_err[~np.isfinite(self.ESDs_err)] = 0
        self.n = (self.ESDs != 0).sum(0)
        self.rbin = (rmax**3-rmin**3)/(rmax**2-rmin**2)*2./3. # area-weighted average

        if(weighted):
            w = (self.ESDs/self.ESDs_err)**2.
            w[~np.isfinite(w)] = 0
            w = w.sum(1)
            self.w = w/np.sum(w, 0)
            self.m_avg = np.sum(self.w*self.m_500)
        else:
            self.w = np.ones([self.size])/self.size

        self.ESD_stat_err = np.zeros(len(bin_edges)-1)* self.ESDs_err.unit
        self.ESDerr = np.zeros(len(bin_edges)-1)* self.ESDs_err.unit
        self.ESD = np.zeros(len(bin_edges)-1)* self.ESDs.unit

        for i in xrange(len(bin_edges)-1):
            self.ESDerr[i] = np.sqrt(np.sum(self.w**2. * self.ESDs_err[:, i]**2.))*self.size/self.n[i]
            self.ESD[i] = np.sum(self.w*self.ESDs[:, i])*self.size/self.n[i]

        self.ESD_stat_err = np.copy(self.ESDerr)* self.ESDerr.unit

        if(cosmic_noise is not None):
            if not bin_edges.unit.is_equivalent('Mpc'):
                print "The Cosmic noise computation is implemented only for \
                        bins in physical units.\
                        If you're stacking using angular distances you are \
                        better off calling cosmic_noise.CLSS() directly."
                return False
            from cosmic_noise import CLSS

            if(type(cosmic_noise) == str):
                self.et_cov = np.load(cosmic_noise)
            else:
                self.et_cov = [None]*self.size

            nbin = len(bin_edges) -1

            if(mscaling):
                self.Cov = np.zeros((nbin, nbin))*(1./u.Mpc/u.Mpc)**2.
            else:
                self.Cov = np.zeros((nbin, nbin))*(u.Msun/u.Mpc/u.Mpc)**2.

            for i in xrange(self.size):
                if(type(cosmic_noise) != str):
                    if(comoving):
                        self.et_cov[i] = CLSS(bin_edges/(1.+self.z[i])/self.da[i],cosmic_noise)
                    else:
                        self.et_cov[i] = CLSS(bin_edges/self.da[i],cosmic_noise)
                for j in xrange(nbin):
                    for k in xrange(nbin):
                            if(mscaling):
                                self.Cov[j, k] += self.et_cov[i][j, k] *\
                                (1./self.da[i]/self.beta_avg[i]/self.m_500[i]*\
                                (const.c**2.)/4./np.pi/const.G/u.rad)**2. * self.w[i]**2. *self.size**2. / self.n[j] / self.n[k]
                            else:
                                self.Cov[j, k] += self.et_cov[i][j, k] * \
                                (1./self.da[i]/self.beta_avg[i] *\
                                (const.c**2.)/4./np.pi/const.G/u.rad)**2. * self.w[i]**2. *self.size**2. / self.n[j] / self.n[k]

            for i in xrange(nbin):
                self.Cov[i, i] += self.ESD_stat_err[i]**2.

            self.ESDerr = np.sqrt(self.Cov.diagonal())


    def __getitem__(self, item):
        return self.clusters[item]

    def __len__(self):
        return self.size


class cluster:
    '''
        Load a source catalog for a cluster and compute tangential and cross component azimuthally averaged shear.

        Parameters
        ----------
        filepath : str
            Path of the source catalog

        rbin : Quantity
            Radial bins to compute the azimuthally averaged shear components around the center of the cluster, can be computed in physical units or in angular units
        gtbin : ndarray
            Tangential components computed using rbin
        gxbin : ndarray
            Cross-components computed using rbin
        dgtbin : ndarray
            Error on the tangential component
    '''

    # Image data
    pixsize = None

    # Cluster data for shear
    xcen, ycen, mmin, mmax = [None for i in xrange(4)]
    da = 0*u.Mpc/u.rad
    z = None

    # Contamination and obscuration parameters
    n_0, r_core, r_max, r_500 = 0*u.arcsec, 0*u.Mpc, 0*u.Mpc, 0*u.Mpc

    def __init__(self, filepath):
        from astropy.table import Table

        self.rbin = None
        self.gtbin = None
        self.gxbin = None
        self.ng = None

        data_table = Table.read(filepath)

        #Load metadata
        if(data_table.meta['PIXUNIT']=='arcsec'):
            self.pixsize = data_table.meta['PIXSIZE'] * u.arcsec

        self.name = data_table.meta['NAME']
        self.sample = data_table.meta['SAMPLE']
        self.filepath = filepath

        #Load columns
        column_list = ['x', 'y', 'm', 'e1', 'e2', 'de', 'pg', 'mu', 'nu', 'delmag', 'e1r', 'e2r']
        for colname in data_table.colnames:
            try:
                i = column_list.index(colname)
                setattr(self, colname, data_table[colname].quantity)
                i+=1
            except ValueError:
                print('Column not recognized: ' + str(colname))
                i+=1
                continue

    def compute_ng(self, bin_edges=None, gridsize=20, comoving=False, maskfile=None):
        '''
            Compute the source density around the cluster center. The results are
            stored in self.nbing (number of galaxies per bin), self.ng (number density),
            rbing (radial bins) and fmg (fraction of area masked per radial bin).
            Takes into account masking.

            Parameters
            ----------
            bin_edges : Quantity or float
                Array or the bin edges. For bins of size N,
                gtbin and gxbin will have size N-1.
            gridsize : int
                Pixel size of the grid used to compute the non-masked area.
                Smaller number = better resolution = slower computation.
            comoving : bool
                Compute using comoving coordinates. Defaults to False
        '''
        #import pyregion
        from region_util import Rutil

        #Magnitude cut
        idx = (self.m > self.mmin) & (self.m < self.mmax) & (self.pg>0.1) & (self.delmag == 0)
        x, y = self.x[idx], self.y[idx]

        # Find distance to closest edge
        max_r = np.array([abs(self.xcen - x.max()), abs(self.xcen - x.min()), abs(self.ycen - y.max()), abs(self.ycen - y.min() )])
        max_r = max_r.min()*self.pixsize

        # Create grid to compute non-masked area
        x_grid, y_grid = np.meshgrid(np.arange(np.ceil(x.max()/gridsize)), np.arange(np.ceil(y.max()/gridsize)))
        x_grid = (x_grid.flatten()+0.5)*gridsize
        y_grid = (y_grid.flatten()+0.5)*gridsize

        #Load mask
        mask = Rutil(maskfile)
        #box = Rutil([x.max()/2., y.max()/2., x.max(), y.max()])

        #Transform x and y in arcsec
        x = self.pixsize*(x - self.xcen)
        y = self.pixsize*(y - self.ycen)
        r = np.sqrt(x**2.+y**2.)

        # Grid in physical quantities
        x_grid_p = self.pixsize*(x_grid - self.xcen)
        y_grid_p = self.pixsize*(y_grid - self.ycen)
        r_grid_p = np.sqrt(x_grid_p**2.+y_grid_p**2.)


        if(bin_edges.unit.is_equivalent('Mpc')):
            r = r * self.da
            r_grid_p = r_grid_p * self.da
            max_r = max_r * self.da
            if(comoving):
                r = r * (1.+self.z)
                r_grid_p = r_grid_p * (1.+self.z)
                max_r = max_r * (1.+self.z)
        else:
            r = r
            r_grid_p = r_grid_p

        rmin = bin_edges[:-1]
        rmax = bin_edges[1:]

        nbin = len(rmin)
        rbin = (rmax**3.-rmin**3.)/(rmax**2.-rmin**2.)*2./3. # area-weighted average
        area = np.pi * (rmax**2. - rmin**2.) # Ring area
        nbing, ng, fmg, dng = [np.zeros(nbin) for i in xrange(4)]
        ng, dng = ng/area.unit, dng/area.unit

        for i in xrange(len(rmin)):
            idx = (r < rmax[i]) & (r >= rmin[i])
            nbing[i] = idx.sum()

            # Reduce area by fraction of points of the grid that are non-masked
            idx_grid = (r_grid_p < rmax[i]) & (r_grid_p >= rmin[i])
            idx_masked = mask(x_grid[idx_grid], y_grid[idx_grid])

            fmg[i] = (~idx_masked).sum()*1./idx_grid.sum()
            ng[i] = nbing[i]/area[i]/fmg[i]
            dng[i] = np.sqrt(nbing[i])/area[i]/fmg[i]

            if(rmax[i]>max_r):
                ng[i] = 0./area[i]
                fmg[i] = 0.
                dng[i] = 0/area[i]


        self.ng = ng
        self.dng = dng
        self.rbing = rbin
        self.nbing = nbing
        self.fmg = fmg

    def compute_shear(self, bin_edges=None, raw=False, contamination=True, comoving=False):
        '''
            Compute tangential and cross component of the shear around the
            cluster center. The results are stored in gtbin and gxbin, the error
            on the tangential shear is stored in dgtbin.

            Does take into account the contamination from cluster members.

            Parameters
            ----------
            bin_edges : Quantity or float
                Array or the bin edges. For bins of size N,
                gtbin and gxbin will have size N-1.
            raw : bool
                Load raw ellipticities instead of the PSF corrected ones.
            contamination : bool
                Compute members contamination and obscuration. Defaults to True
            comoving : bool
                Compute using comoving coordinates. Defaults to False
        '''


        idx = (self.m > self.mmin) & (self.m < self.mmax) & (self.pg>0.1) & (self.delmag == 0)

        x, y, pg, de, m, mu = self.x[idx], self.y[idx], self.pg[idx], self.de[idx], self.m[idx], self.mu[idx]

        if(raw):
            e1, e2 =  self.e1r[idx], self.e2r[idx]
            self.raw = True
        else:
            e1, e2 =  self.e1[idx], self.e2[idx]
            self.raw = False

        #Transform x and y in arcsec
        x = self.pixsize*(x - self.xcen)
        y = self.pixsize*(y - self.ycen)
        r = np.sqrt(x**2.+y**2.)
        #pg[pg<0.] = 0


        #weights
        w = pg**2./((0.25*pg)**2. + de**2.)
        wg = pg/((0.25*pg)**2. + de**2.)
        et = -((x*x-y*y)*e1 + 2.0*x*y*e2)/r**2.
        ex = ((x*x-y*y)*e2-2.0*x*y*e1)/r**2.

        bin_edges = Quantity(bin_edges)

        if(bin_edges.unit.is_equivalent('Mpc')):
            r = r * self.da
            if(comoving):
                r = r* (1.+self.z)
        else:
            r = r

        rmin = bin_edges[:-1]
        rmax = bin_edges[1:]

        nbin = len(rmin)
        rbin = (rmax**3.-rmin**3.)/(rmax**2.-rmin**2.)*2./3. # area-weighted average
        nbin, gtbin, gxbin, dgtbin, kbin = [np.zeros(nbin) for i in xrange(5)]

        rbin_cont = rbin
        if(comoving):
            # contamination should be evaluated using physical distances, NOT comoving
            rbin_cont = rbin/(1.+self.z)

        for i in xrange(len(rmin)):
            idx = (r < rmax[i]) & (r >= rmin[i])
            nbin[i] = idx.sum()
            gtbin[i] = np.sum(wg[idx]*et[idx])/np.sum(w[idx])
            gxbin[i] = np.sum(wg[idx]*ex[idx])/np.sum(w[idx])
            dgtbin[i] = np.sqrt(1./np.sum(w[idx]))
            kbin[i] = np.sum(mu[idx]*w[idx])/np.sum(w[idx])

        gtbin = gtbin/kbin
        gxbin = gxbin/kbin

        if(contamination):
            if(bin_edges.unit.is_equivalent('Mpc')):
                fcontam = self.n_0*self.da * (1./(rbin_cont + self.r_core) - 1./(self.r_max + self.r_core))
                fcontam[rbin>self.r_max] = 0
                fobscured = 1+0.022/(0.14+(rbin_cont/self.r_500)**2.)
            else:
                fcontam = self.n_0 * (1./(rbin_cont + self.r_core/self.da) - 1./(self.r_max/self.da + self.r_core/self.da))
                fcontam[rbin>self.r_max/self.da] = 0
                fobscured = 1.+0.022/(0.14+(rbin_cont/self.r_500*self.da)**2.)

            self.gtbin = (gtbin*(fcontam*fobscured + 1)).to(1)
            self.gxbin = gxbin
            self.dgtbin = (dgtbin*(fcontam*fobscured + 1)).to(1)
            self.fcontam = (fcontam*fobscured).to(1)
        else:
            self.gtbin = gtbin
            self.gxbin = gxbin
            self.dgtbin = dgtbin
            self.fcontam = None

        self.rbin = rbin
        self.nbin = nbin

    def print_info(self):
        '''
            Print cluster info
        '''
        print 'Cluster name: '+self.name
        print 'Source catalog: '+self.filepath
        print 'Pixel position of cluster center: '+str(self.xcen)+', '+str(self.ycen)
        print 'Contamination parameters (n_0, r_core, r_max, r_500): '+str(n_0)+' '+str(r_core)+' '+str(r_max)+' '+str(r_500)
        print 'Angular diameter distance (h_100=1, O_m = 0.3, O_L = 0.7): '+str(self.da)
        print 'Magnitude range: ('+str(mmin)+', '+str(mmax)+')'

    def plot_sources(self, radius=None):
        '''
            Plot the source catalog and plots concentric circles.

            Parameters
            ----------
                radius : Quantity or nd.array or float
                    radius (or a set of radii) of the circle centered on the
                    cluster center. Allowed units are length, angle or pixel size.
        '''
        from matplotlib import pyplot as plt

        plt.suptitle(self.name+r'  $r_{500}$ = '+str(self.r_500)+' h$^{-1}$')
        if(radius is not None):
            radius = Quantity(radius)
            if(radius.unit.is_equivalent('1')):
                # Radius is in pixel
                radius = radius.value
            if(radius.unit.is_equivalent('arcsec')):
                # Radius is in arcsec
                radius = radius.to('arcsec')
                x, y, xc, yc = (self.x*self.pixsize).to('arcsec').value, \
                            (self.y*self.pixsize).to('arcsec').value, \
                            (self.xcen*self.pixsize).to('arcsec').value, \
                            (self.ycen*self.pixsize).to('arcsec').value
                unit = 'arcsec'
            if(radius.unit.is_equivalent('Mpc')):
                #Radius is in h-1 Mpc
                radius = radius.to('Mpc').value
                x, y, xc, yc = (self.x*self.da*self.pixsize).to('Mpc').value, \
                            (self.y*self.da*self.pixsize).to('Mpc').value, \
                            (self.xcen*self.da*self.pixsize).to('Mpc').value, \
                            (self.ycen*self.da*self.pixsize).to('Mpc').value
                unit = 'h-1 Mpc'

            plt.scatter(x, y, marker='.', c='k', zorder=1)
            plt.scatter(xc, yc, marker='*', c='r', s=100, zorder=100)
            try:
                circle = [None]*len(radius)
                for i in xrange(len(radius)):
                    circle[i] = plt.Circle((xc, yc), radius[i], color='r', fill=False, zorder=200)
                    plt.gca().add_artist(circle[i])
            except TypeError:
                circle = plt.Circle((xc, yc), radius, color='r', fill=False, zorder=200)
                plt.gca().add_artist(circle)

            plt.gca().set_xlabel('x ('+unit+')')
            plt.gca().set_ylabel('y ('+unit+')')

        else:

            plt.scatter(self.x, self.y, marker='.', c='k', zorder=1)
            plt.scatter(self.xcen, self.ycen, marker='*', c='r', s=100, zorder=100)
            plt.gca().set_xlabel('x')
            plt.gca().set_ylabel('y')
