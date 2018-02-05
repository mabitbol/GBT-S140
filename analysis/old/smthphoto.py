import numpy as np
import healpy as hp

kboltz=1.3806503e-23 
clight=299792458.0 
hplanck=6.626068e-34 
TCMB = 2.72548 
d2r = np.pi / 180.
pi = np.pi

lonc = 107.2
latc = 5.2

## These maps are smoothed to 1 degree and in units of Jy / pixel.

wmapfiles = ['../externaldata/wmap_smoothed_22.fits', \
             '../externaldata/wmap_smoothed_32.fits',\
             '../externaldata/wmap_smoothed_40.fits',\
             '../externaldata/wmap_smoothed_60.fits',\
             '../externaldata/wmap_smoothed_93.fits']
wmapfreqs = np.array([22.71, 32.95, 40.65, 60.64, 93.44]) * 1e9


planckfiles = ['../externaldata/planck_smoothed_28.fits', \
               '../externaldata/planck_smoothed_44.fits', \
               '../externaldata/planck_smoothed_70.fits', \
               '../externaldata/planck_smoothed_143.fits', \
               '../externaldata/planck_smoothed_217.fits', \
               '../externaldata/planck_smoothed_353.fits', \
               '../externaldata/planck_smoothed_545.fits', \
               '../externaldata/planck_smoothed_857.fits']
planckfreqs = np.array([28.4, 44.1, 70.4, 143., 217., 353., 545., 857.]) * 1e9
planckbeams = np.array([32.3, 27.1, 13.3, 7.3, 5., 4.8, 4.7, 4.3])


irisfiles = ['../externaldata/iris_smoothed_25.fits', \
             '../externaldata/iris_smoothed_12.fits', \
             '../externaldata/iris_smoothed_5.fits', \
             '../externaldata/iris_smoothed_3.fits']
irisfreqs = np.array([25000, 12000, 5000, 3000]) * 1.e9
irisbeams = np.array([3.8, 3.8, 4.0, 4.3])

dirbefiles = ['../externaldata/dirbe_smoothed_240.fits', \
              '../externaldata/dirbe_smoothed_136.fits', \
              '../externaldata/dirbe_smoothed_85.fits', \
              '../externaldata/dirbe_smoothed_61.fits', \
              '../externaldata/dirbe_smoothed_25.fits', \
              '../externaldata/dirbe_smoothed_12.fits', \
              '../externaldata/dirbe_smoothed_5.fits', \
              '../externaldata/dirbe_smoothed_3.fits', \
              '../externaldata/dirbe_smoothed_2.fits', \
              '../externaldata/dirbe_smoothed_1.fits']
dirbefreqs = np.array([240, 136.36, 85.71, 61.22, 25, 12, 5, 3, 2.1428, 1.25]) * 1.e12
dirbebeams = np.array([37.6, 41.0, 39.0, 41.5, 41.0, 41.5, 42.3, 41.0, 40.4, 39.5])



def get_aperture(nside=512, lonc=107.2, latc=5.2, inside=80, outside=100, radius=60):
    vecc = hp.rotator.dir2vec(lonc, latc, lonlat=True)
    rmask = hp.query_disc(nside, vecc, (radius/60.)*d2r, inclusive=False)
    amaskout = set(hp.query_disc(nside, vecc, (outside/60.)*d2r, inclusive=False))
    amaskin = set(hp.query_disc(nside, vecc, (inside/60.)*d2r, inclusive=False))
    amask = np.array(list(amaskout.difference(amaskin)))
    return rmask, amask

def get_flux(xmap, rmask, amask):
    rdata = y[rmask]
    adata = y[amask]
    flux = np.sum(rdata - np.median(adata))
    rms = np.std(adata) * np.sqrt(len(rdata) + pi/2. * float(len(rdata)**2) / len(adata))
    return flux, rms

def experiment_flux(filelist, ):
    rmask, amask = get_
    for fl in filelist:
        



def wmap_calc(wmap, nu, inside=60., outside=90., radius=60.):
    wmapmap = hp.read_map(wmap, verbose=False) * 1.e-3
    x = np.copy(wmapmap)
    
    nside = hp.get_nside(x)
    vecc = hp.rotator.dir2vec(lonc, latc, lonlat=True)
    rmask = hp.query_disc(nside, vecc, (radius/60.)*d2r, inclusive=False)
    amaskout = set(hp.query_disc(nside, vecc, (outside/60.)*d2r, inclusive=False))
    amaskin = set(hp.query_disc(nside, vecc, (inside/60.)*d2r, inclusive=False))
    amask = np.array(list(amaskout.difference(amaskin)))
    
    X = hplanck * nu / (kboltz * TCMB) 
    kthermo_to_intensity = 2. * kboltz * (nu / clight)**2 * (X**2 * np.exp(X)) / (np.exp(X) - 1.)**2
    y = x * kthermo_to_intensity * hp.nside2pixarea(nside) * 1.e26 
    
    rdata = y[rmask]
    adata = y[amask]
    flux = np.sum(rdata - np.median(adata))
    rms = np.std(adata) * np.sqrt(len(rdata) + pi/2. * float(len(rdata)**2) / len(adata))
    return flux, np.sqrt(rms**2 + (0.03*flux)**2)

