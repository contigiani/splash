from astropy import units as u
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import simps

'''
    NFW profile
'''

def NFW(R, params, h_max=40, epsr=0.00001):
    R_min = 0.1
    sigmatemp = lambda x: x*NFW_S(x, params, h_max)
    result = np.zeros(len(R))
    mu = integrate.quad(sigmatemp, 0.001, R_min, epsrel=epsr)[0]

    for i in xrange(len(R)):
        result[i] = 2.*(mu+integrate.quad(sigmatemp, R_min, R[i], epsrel=epsr)[0])/R[i]/R[i] - NFW_S(R[i], params, h_max)

    return result

def NFW_S(r, params, h_max=40):
    r = np.atleast_1d(r)
    result = np.zeros(len(r))

    for i in xrange(len(r)):
        rho_temp = lambda h: rho_NFW(np.sqrt(r[i]**2.+h**2.), params)
        result[i] = 2*integrate.quad(rho_temp, 0., h_max)[0]

    return result

def rho_NFW(r, params):
    rs, rhos = params

    return rhos/(r/rs)/(1.+r/rs)**2.



'''
    DK14 profile, see Diemer&Kravtsov 2014, Baxter+ 2017
'''


def DK14(R, params, h_max=40., R_min=0.05, mode=0, epsr=0.001, hp=False):
    '''
        Return the excess surface density at R (in h100-1 Mpc)

        Parameters
        ----------
        mode : int
            One of 0, 1, 2, 3. Corresponding to:
            0: yes infall, yes f_trans (requires 8 parameters)
            1: no infall, no f_trans (requires 5 parameters)
        hp : bool
            forces high precision quad integral with relative error epsr

    '''
    sigmatemp = lambda x: x*DK14_S(x, params, h_max, mode, hp)
    result = np.zeros(len(R))
    mu = integrate.quad(sigmatemp, 0.0001, R_min, epsrel=epsr)[0]

    integral = 0
    for i in xrange(len(R)):
        if(hp):
            integral = integrate.quad(sigmatemp, R_min, R[i], epsrel=epsr)[0]
        else:
            if(i==0):
                int_x = np.linspace(R_min, R[i], 10)
            else:
                int_x = np.linspace(R[i-1], R[i], 10)
            integral += simps(int_x*DK14_S(int_x, params, h_max, mode), int_x)
        result[i] = 2.*(mu+integral)/R[i]/R[i] - DK14_S(R[i], params, h_max, mode, hp)

    return result

def DK14_S(r, params, h_max=40, mode=0, hp=False):
    '''
        Returns the line of sight integral of the 3D profile between -hmax and +hmax, evaluated at r.
    '''
    r = np.atleast_1d(r)
    result = np.zeros(len(r))

    for i in xrange(len(r)):
        if(hp):
            rho_temp = lambda h: rho(np.sqrt(r[i]**2.+h**2.), params, mode)
            integral = integrate.quad(rho_temp, 0., h_max)[0]
        else:
            int_h = np.linspace(0, h_max, 200)
            integral = simps(rho(np.sqrt(r[i]**2.+int_h**2.), params, mode), int_h)
        result[i] = 2*integral

    return result

def rho(r, params, mode=0):
    if(mode == 0):
        rho_s, r_s, logalpha, r_t, logbeta, loggamma, rho_0, s_e  = params
        return rho_Ein(r, [rho_s, r_s, logalpha])*f_trans(r, [r_t, logbeta, loggamma])+rho_infall(r, [rho_0, s_e])
    if(mode == 1):
        if(len(params)==8):
            rho_s, r_s, logalpha, r_t, logbeta, loggamma, rho_0, s_e  = params
        else:
            rho_s, r_s, logalpha, rho_0, s_e = params
        return rho_Ein(r, [rho_s, r_s, logalpha])#+rho_infall(r, [rho_0, s_e])

def rho_Ein(r, params):
    rho_s, r_s, logalpha = params
    alpha = (10.**logalpha)
    return rho_s * np.exp(-2./alpha*( (r/r_s)**alpha-1.) )

def f_trans(r, params):
    r_t, logbeta, loggamma = params
    beta = (10.**logbeta)
    gamma = (10.**loggamma)
    return (1.+(r/r_t)**beta)**(-gamma/beta)

def rho_infall(r, params):
    rho_0, s_e = params
    r_0 = 1.5
    return rho_0*(r/r_0)**(-s_e)
