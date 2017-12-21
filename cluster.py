
class cluster_sanple:
    def __init__(self, filename):

        cluster_data = Table.read(filename, format='ascii')
        self.size = len(cluster_data)

        column_names = ['xcen', 'ycen', 'Mpc', 'mmin', 'mmax', 'name_cl']
        xcen, ycen, mpc, mmin, mmax, cluster_name = [cluster_data[string][cluster_id] for string in column_names]

        for i in xrange(N):
            self.cluster[i] = cluster(names[i], dirname+names[i]+'.dat')

    def stack_ESD(self, bins=None):
        return True



class cluster:
    def __init__(self):
        self.rbin = None
        self.gtbin = None
        self.gxbin = None
        return True


    def gt(self, bins=None):
        '''
            Return tangential component of the shear around the cluster.

            Parameters
            ----------
            bins: Quantity
                bin array specifying the bins to compute gt.
                Depending on the units of bin, it is projected in the sky or in comoving distance with h = 1

            Returns
            -------
            gt : np.ndarray
                Array of size N-1, where N is the size of bin.
        '''
        if(bins==None) and (self.gtbin != None):
            return self.gtbin

    def gx(self, bins=None):
        '''
            Return cross component of the shear around the cluster.

            Parameters
            ----------
            bins: Quantity
                bin array specifying the bins to compute gx.
                Depending on the units of bin, it is projected in the sky or in comoving distance with h = 1

            Returns
            -------
            gt : np.ndarray
                Array of size N-1, where N is the size of bin.

        '''
        if(bins==None) and (self.gxbin != None):
            return self.gxbin

    def print_info(self):
        print 'Cluster name: '+self.name
        print 'Source catalog: '+self.filename
