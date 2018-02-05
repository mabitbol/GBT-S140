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

wmapfiles = ['../externaldata/wmap_band_smth_deconv_imap_r9_9yr_K_v5.fits', \
             '../externaldata/wmap_band_smth_deconv_imap_r9_9yr_Ka_v5.fits',\
             '../externaldata/wmap_band_smth_deconv_imap_r9_9yr_Q_v5.fits', \
             '../externaldata/wmap_band_smth_deconv_imap_r9_9yr_V_v5.fits', \
             '../externaldata/wmap_band_smth_deconv_imap_r9_9yr_W_v5.fits']
wmapfreqs = np.array([22.71, 32.95, 40.65, 60.64, 93.44]) * 1e9


planckfiles = ['../externaldata/LFI_SkyMap_030-field-IQU_1024_R2.01_full.fits', \
              '../externaldata/LFI_SkyMap_044-field-IQU_1024_R2.01_full.fits', \
              '../externaldata/LFI_SkyMap_070-field-IQU_1024_R2.01_full.fits', \
              '../externaldata/HFI_SkyMap_143-field-IQU_2048_R2.02_full.fits', \
              '../externaldata/HFI_SkyMap_217-field-IQU_2048_R2.02_full.fits', \
              '../externaldata/HFI_SkyMap_353-field-IQU_2048_R2.02_full.fits', \
              '../externaldata/HFI_SkyMap_545-field-Int_2048_R2.02_full.fits', \
              '../externaldata/HFI_SkyMap_857-field-Int_2048_R2.02_full.fits']
planckfreqs = np.array([28.4, 44.1, 70.4, 143., 217., 353., 545., 857.]) * 1e9
planckbeams = np.array([32.3, 27.1, 13.3, 7.3, 5., 4.8, 4.7, 4.3])


irisfiles = ['../externaldata/IRIS_nohole_1_2048.fits', \
            '../externaldata/IRIS_nohole_2_2048.fits', \
            '../externaldata/IRIS_nohole_3_2048.fits', \
            '../externaldata/IRIS_nohole_4_2048.fits']
irisfreqs = np.array([25000, 12000, 5000, 3000]) * 1.e9
irisbeams = np.array([3.8, 3.8, 4.0, 4.3])

dirbefiles = ['../externaldata/DIRBE_ZSMA_1_256.fits', \
              '../externaldata/DIRBE_ZSMA_2_256.fits', \
              '../externaldata/DIRBE_ZSMA_3_256.fits', \
              '../externaldata/DIRBE_ZSMA_4_256.fits', \
              '../externaldata/DIRBE_ZSMA_5_256.fits', \
              '../externaldata/DIRBE_ZSMA_6_256.fits', \
              '../externaldata/DIRBE_ZSMA_7_256.fits', \
              '../externaldata/DIRBE_ZSMA_8_256.fits', \
              '../externaldata/DIRBE_ZSMA_9_256.fits', \
              '../externaldata/DIRBE_ZSMA_10_256.fits']
dirbefreqs = np.array([240, 136.36, 85.71, 61.22, 25, 12, 5, 3, 2.1428, 1.25]) * 1.e12
dirbebeams = np.array([37.6, 41.0, 39.0, 41.5, 41.0, 41.5, 42.3, 41.0, 40.4, 39.5])


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
    return int(nu*1e-9), flux, np.sqrt(rms**2 + (0.03*flux)**2)

def planck_calc(planck, nu, beam, inside=60., outside=90., radius=60.):
    planckmap = hp.read_map(planck, verbose=False)
    newbeam = np.sqrt(60.**2 - beam**2) / 60.
    planckmap = hp.ud_grade(hp.smoothing(planckmap, fwhm=newbeam*d2r, verbose=False), 512)
    
    x = np.copy(planckmap)
    nside = hp.get_nside(x)
    vecc = hp.rotator.dir2vec(lonc, latc, lonlat=True)
    rmask = hp.query_disc(nside, vecc, (radius/60.)*d2r, inclusive=False)
    amaskout = set(hp.query_disc(nside, vecc, (outside/60.)*d2r, inclusive=False))
    amaskin = set(hp.query_disc(nside, vecc, (inside/60.)*d2r, inclusive=False))
    amask = np.array(list(amaskout.difference(amaskin)))
    
    X = hplanck * nu / (kboltz * TCMB) 
    kthermo_to_intensity = 2. * kboltz * (nu / clight)**2 * (X**2 * np.exp(X)) / (np.exp(X) - 1.)**2
    if nu < 400e9:
        y = x * kthermo_to_intensity * hp.nside2pixarea(nside) * 1.e26 
    else:
        y = x * 1e6 * hp.nside2pixarea(nside)
    rdata = y[rmask]
    adata = y[amask]
    flux = np.sum(rdata - np.median(adata))
    rms = np.std(adata) * np.sqrt(len(rdata) + pi/2. * float(len(rdata)**2) / len(adata))
    return int(nu*1e-9), flux, np.sqrt(rms**2 + (0.03*flux)**2)

def iris_flux(iris, beam=4., inside=60., outside=90., radius=60.):
    z = hp.read_map(iris)
    mask = z == -32768
    z[mask] = hp.UNSEEN
    mask = np.isnan(z)
    z[mask] = hp.UNSEEN
    newbeam = np.sqrt(60.**2 - beam**2) / 60.
    z = hp.ud_grade(hp.smoothing(z, fwhm=newbeam*np.pi/180., verbose=False), 512)
    
    nside = hp.get_nside(z)
    vecc = hp.rotator.dir2vec(lonc, latc, lonlat=True)
    rmask = hp.query_disc(nside, vecc, (radius/60.)*d2r, inclusive=False)
    amaskout = set(hp.query_disc(nside, vecc, (outside/60.)*d2r, inclusive=False))
    amaskin = set(hp.query_disc(nside, vecc, (inside/60.)*d2r, inclusive=False))
    amask = np.array(list(amaskout.difference(amaskin)))
    
    y = z * hp.nside2pixarea(nside) * 1.e6
    
    rdata = y[rmask]
    adata = y[amask]
    flux = np.sum(rdata - np.median(adata))
    rms = np.std(adata) * np.sqrt(len(rdata) + pi/2. * float(len(rdata)**2) / len(adata))
    return flux, rms

def dirbe_flux(dirbe, beam=40., inside=60., outside=90., radius=60.):
    z = hp.read_map(dirbe)
    newbeam = np.sqrt(60.**2 - beam**2) / 60.
    z = hp.ud_grade(hp.smoothing(z, fwhm=newbeam*np.pi/180., verbose=False), 512)
    
    nside = hp.get_nside(z)
    vecc = hp.rotator.dir2vec(lonc, latc, lonlat=True)
    rmask = hp.query_disc(nside, vecc, (radius/60.)*d2r, inclusive=False)
    amaskout = set(hp.query_disc(nside, vecc, (outside/60.)*d2r, inclusive=False))
    amaskin = set(hp.query_disc(nside, vecc, (inside/60.)*d2r, inclusive=False))
    amask = np.array(list(amaskout.difference(amaskin)))
    
    y = z * hp.nside2pixarea(nside) * 1.e6
    
    rdata = y[rmask]
    adata = y[amask]
    flux = np.sum(rdata - np.median(adata))
    rms = np.std(adata) * np.sqrt(len(rdata) + pi/2. * float(len(rdata)**2) / len(adata))
    return flux, rms
