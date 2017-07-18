import numpy as np
from numpy import log10
from scipy import interpolate

hplanck = 6.626068e-34  # MKS
kboltz = 1.3806503e-23  # MKS

def synch_rad(nu, As=288., alps=-0.82, w2s=0.2):
    nu0s = 100.e9
    return As * (nu / nu0s) ** alps * (1. + 0.5 * w2s * np.log(nu / nu0s) ** 2)

def synch_rad1(nu, As=288., alps=-0.82):
    nu0s = 100.e9
    return As * (nu / nu0s) ** alps

def freefree_rad(nu, EM=1.6, Te=140.):
    Teff = (Te / 1.e3) ** (3. / 2)
    nuff = 255.33e9 * Teff
    gff = 1. + np.log(1. + (nuff / nu) ** (np.sqrt(3) / np.pi))
    return EM * gff

def spinning_dust(nu, Asd=1.):
    ame_file = np.load('/home/mabitboo/GBT-S140/externaldata/ame.npy').astype(ndp)
    ame_nu = ame_file[0]
    ame_I = ame_file[1]
    fsd = interpolate.interp1d(log10(ame_nu), log10(ame_I), bounds_error=False, fill_value="extrapolate")
    return Asd * 10.**fsd(log10(nu)) * 1.e26

def thermal_dust_rad(nu, Ad=1.36e6, Bd=1.53, Td=21.):
    X = hplanck * nu / (kboltz * Td)
    return Ad * X**Bd * X**3. / (np.exp(X) - 1.0)

def cib_rad(nu, Acib=3.46e5, Bcib=0.86, Tcib=18.8):
    X = hplanck * nu / (kboltz * Tcib)
    return Acib * X**Bcib * X**3. / (np.exp(X) - 1.0)

def co_rad(nu, Aco=1.):
    x = np.load('/home/mabitboo/GBT-S140/externaldata/co_arrays.npy').astype(ndp)
    freqs = x[0]
    co = x[1]
    fs = interpolate.interp1d(log10(freqs), log10(co), bounds_error=False, fill_value="extrapolate")
    return Aco * 10. ** fs(log10(nu))


def dust_moments(nu, Adm=3.2e-4, alphadm=1.22, Tdm=21.1, omega1=0.09):
    X = hplanck * nu / (kboltz * Tdm)
    nu0 = 100.e9
    lnnu = np.log(nu/nu0)
    Y1 = X * np.exp(X) / (np.exp(X) - 1.)
    zeroth = Adm * (nu/nu0)**alphadm * nu**3 / (np.exp(X) - 1.)
    return zeroth * (1. + 0.5 * omega1 * lnnu**2)


