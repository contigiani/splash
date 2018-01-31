from astropy import units as u
import numpy as np
import scipy.integrate as integrate

'''
    DK14 profile, see Diemer&Kravtsov 2014, Baxter+ 2017
'''


def DK14(r, params, h_max, mode=0):
    '''
        Returns the line of sight integral of the 3D profile between -hmax and +hmax, evaluated at r.

        Parameters
        ----------
        mode : int
            One of 0, 1, 2, 3. Corresponding to:
            0: yes infall, yes f_trans (requires 8 parameters)
            1: yes infall, no f_trans (requires 5 parameters)
            2: no infall, yes f_trans (requires 6 parameters)
            3: no infall, no f_trans (requires 3 parameters)

    '''
    result = np.zeros(len(r))

    for i in xrange(len(r)):
        rho_temp = lambda h: rho(np.sqrt(r[i]**2.+h**2.), params, mode)
        result[i] = integrate.quad(rho_temp, -h_max, h_max)[0]

    return result

def rho(r, params, mode):
    if(mode == 0):
        rho_s, r_s, logalpha, r_t, logbeta, loggamma, rho_0, s_e  = params
        return rho_Ein(r, [rho_s, r_s, logalpha])*f_trans(r, [r_t, logbeta, loggamma])+rho_infall(r, [rho_0, s_e])
    if(mode == 1):
        rho_s, r_s, logalpha, rho_0, s_e = params
        return rho_Ein(r, [rho_s, r_s, logalpha])+rho_infall(r, [rho_0, s_e])
    if(mode==2):
        rho_s, r_s, logalpha, r_t, logbeta, loggamma = params
        return rho_Ein(r, [rho_s, r_s, logalpha])*f_trans(r, [r_t, logbeta, loggamma])
    if(mode==3):
        rho_s, r_s, logalpha = params
        return rho_Ein(r, [rho_s, r_s, logalpha])

def rho_Ein(r, params):
    rho_s, r_s, logalpha = params
    alpha = (10.**logalpha)
    return rho_s * np.exp(-2./alpha*( (r/r_s)**alpha-1.) )

def f_trans(r, params):
    r_t, logbeta, loggamma = params
    beta = alpha = (10.**logbeta)
    gamma = alpha = (10.**loggamma)
    return (1.+(r/r_t)**beta)**(-gamma/beta)

def rho_infall(r, params):
    rho_0, s_e = params
    r_0 = 1.5

    return rho_0*(r/r_0)**(-s_e)
