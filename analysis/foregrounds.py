import numpy as np
import math, os
import glob
from scipy import interpolate
from scipy import integrate
from scipy import special
from astropy.io import fits

TCMB = 2.726 #Kelvin
hplanck=6.626068e-34 #MKS
kboltz=1.3806503e-23 #MKS
clight=299792458.0 #MKS
m_elec = 510.999 #keV!

def krj_to_radiance(nu, y):
    return 2.0 * nu*nu /(clight**2) * kboltz * y

def blackbody(nu, T):
    X  = hplanck * nu / (kboltz * T)
    return 2.0 * hplanck * (nu*nu*nu) / (clight**2) * (1.0 / (np.exp(X) - 1.0))

def dbdt(nu, T):
    return 2.0 * (X*X*X*X) * np.exp(X) * (kboltz*T)**3 / (hplanck * clight)**2 / (np.exp(X) - 1.0)**2

def thermal_dust(nu, Ad=163.0e-6, Bd=1.51, Td=21.0):
    nu0 = 545.0e9   #planck frequency
    gam = hplanck/(kboltz*Td)   
    return Ad * (nu/nu0)**(Bd+1.0) * (np.exp(gam*nu0) - 1.0) / (np.exp(gam*nu) - 1.0)

def synchrotron(nu, As=20.0, alpha=0.26):
    nu0 = 408.0e6                                     # Hz
    synch_temp = fits.open('../externaldata/COM_CompMap_Synchrotron-commander_0256_R2.00.fits')
    synch_nu = synch_temp[2].data.field(0)          # GHz
    synch_nu *= 1.e9                                # Hz
    synch_I = synch_temp[2].data.field(1)           # W/Hz/sr/m^2
    fs = interpolate.interp1d(np.log10(synch_nu), np.log10(synch_I))
    numer_fs = 10.0**fs(np.log10(nu/alpha))
    denom_fs = 10.0**fs(np.log10(nu0/alpha))
    return As * (nu0/nu)**2 * numer_fs / denom_fs

def freefree(nu, EM=13.0, Te=7000.0):
    T4 = (Te * 10**-4)**(-3./2.)
    f9 = nu / (10**9)
    gff = np.log10(np.exp(5.960 - (np.sqrt(3.)/np.pi) * np.log10(f9*T4)) + np.e)
    tau = 0.05468 * (Te**(-3./2.)) * EM * gff / f9**2
    return (1.0 - np.exp(-tau)) * Te

def ame(nu, Asd=92.0e-6, nup=19.0e9, nu0=22.8e9):
    # template nu go from 50 MHz to 500 GHz...
    # nu0 = 22.8e9 or 41.0e9
    # Asd2 = 18
    nup0 = 30.0e9
    ame_temp = fits.open('../externaldata/COM_CompMap_AME-commander_0256_R2.00.fits')
    ame_nu = ame_temp[3].data.field(0)
    ame_nu *= 1.e9                      # Hz 
    ame_I = ame_temp[3].data.field(1)   # Jy cm^2 /sr/H
    ame_I *= 1.0e26
    fsd = interpolate.interp1d(np.log10(ame_nu), np.log10(ame_I), bounds_error=False, fill_value=1.e-6)
    numer_fsd = 10.0**fsd(np.log10(nu*nup0/nup))
    denom_fsd = 10.0**fsd(np.log10(nu0*nup0/nup))
    return Asd * (nu0/nu)**2 * numer_fsd / denom_fsd
    
def sz(nu, ysz=1.4e-6):
    X = hplanck*nu/(kboltz*TCMB)
    gf = (np.exp(X)-1)**2 / (X*X*np.exp(X))
    return (ysz*10**6)*TCMB * ( (X*np.exp(X)+1.)/(np.exp(X)-1.) - 4.) / gf

def cmb(freqs, T=TCMB, A=3.0e-6):
    X = hplanck*freqs/(kboltz*T)
    gf = (np.exp(X)-1)**2 / (X*X*np.exp(X))
    return A/gf
    

