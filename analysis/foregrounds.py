import numpy as np
from scipy import interpolate

TCMB = 2.7255
hplanck=6.626068e-34 
kboltz=1.3806503e-23 
clight=299792458.0 
jy = 1.e26

def thermal_dust(nu, Ad=163.0e-6, Bd=1.51, Td=21.0):
    nu0 = 545.0e9
    gam = hplanck / (kboltz * Td)   
    conv = jy * 2.0 * nu * nu /clight**2 * kboltz
    return conv * Ad * (nu / nu0)**(Bd + 1.) * (np.exp(gam * nu0) - 1.) / (np.exp(gam * nu) - 1.)

def ame(nu, Asd=92.0e-6, nup=19.0e9):
    nu0 = 22.8e9
    nup0 = 30.0e9
    amefile = np.load('../externaldata/ame.npy')
    ame_nu = amefile[0, :]
    ame_I = amefile[1, :]
    fsd = interpolate.interp1d( np.log(ame_nu), np.log(ame_I), bounds_error=False, fill_value='extrapolate')
    numer_fsd = fsd( np.log( nu * nup0 / nup ))
    denom_fsd = fsd( np.log( nu0 * nup0 / nup ))
    conv = jy * 2.0 * nu * nu /clight**2 * kboltz
    return conv * Asd * (nu0/nu)**2 * np.exp(numer_fsd - denom_fsd)

def cmb(nu, A=3.0e-6):
    X = hplanck * nu / (kboltz * TCMB)
    gf = (np.exp(X)-1)**2 / (X*X*np.exp(X))
    conv = jy * 2.0 * nu * nu /clight**2 * kboltz
    return conv * A / gf

def freefreep(nu, EM=100., Te=8000.):
    nu9 = nu * 1.e-9
    gff = np.log(4.955e-2 / nu9) + 1.5 * np.log(Te)
    tff = 3.014e-2 * (Te**-1.5) * (nu9**-2) * EM * gff
    conv = jy * 2.0 * nu * nu /clight**2 * kboltz
    return conv * Te * (1. - np.exp(-tff))


def synchrotron(nu, As=200, alps=-0.8):
    nu0s = 100.e9
    return As * (nu / nu0s)**alps
