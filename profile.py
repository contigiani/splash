from astropy import units as u
import numpy as np
import scipy.integrate as integrate

'''
    DK14 profile, see Diemer&Kravtsov 2014, Baxter+ 2017
'''


def DK14(r, params, h_max):
    '''
        Returns the line of sight integral of the 3D profile between -hmax and +hmax, evaluated at r. 
    '''
    result = np.zeros(len(r))

    for i in xrange(len(r)):
        rho_temp = lambda h: rho(np.sqrt(r[i]**2.+h**2.), params)
        result[i] = integrate.quad(rho_temp, -h_max, h_max)[0]

    return result

def rho(r, params):
    rho_s, r_s, alpha, r_t, beta, gamma, rho_0, s_e = params

    return rho_Ein(r, [rho_s, r_s, alpha])*f_trans(r, [r_t, beta, gamma])+rho_infall(r, [rho_0, s_e])

def rho_Ein(r, params):
    rho_s, r_s, alpha = params

    return rho_s * np.exp(-2./alpha*( (r/r_s)**alpha-1.) )

def f_trans(r, params):
    r_t, beta, gamma = params

    return (1.+(r/r_t)**beta)**(-gamma/beta)

def rho_infall(r, params):
    rho_0, s_e = params
    r_0 = 1.5

    return rho_0*(r/r_0)**(-s_e)
