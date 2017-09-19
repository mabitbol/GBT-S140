import numpy as np
from scipy.ndimage.filters import gaussian_filter
from astropy import coordinates
from astropy import units as u
import healpy as hp
from numpy import linalg as linalg

kboltz=1.3806503e-23 #MKS
clight=299792458.0 #MKS
hplanck=6.626068e-34 #MKS
TCMB = 2.72548 #Kelvin
d2r = np.pi / 180.

def naive_map(data,ra,dec,ra_num_bins=60,dec_num_bins=50):
    ra_bins = np.linspace(ra.min(),ra.max(),ra_num_bins)
    dec_bins = np.linspace(dec.min(),dec.max(),dec_num_bins)
    p,_,_ = np.histogram2d(ra,dec,bins=(ra_bins,dec_bins),weights=data)
    hits,_,_ = np.histogram2d(ra,dec,bins=(ra_bins,dec_bins))
    return p.T, hits.T, ra_bins, dec_bins

def aperture_photometry_gbt(bank='A', session='5', rc=107.2, dc=5.2, rad=1., pixbeam=1., smth=None, removeplane=True):
    dataf = np.load('/home/mabitbol/GBT-S140/datamaps/tod'+bank+'_'+session+'.npz')
    tmask = dataf['tmask']
    ras = dataf['ras'][tmask]
    decs = dataf['decs'][tmask]
    calibrated = dataf['calibrated'][tmask]
    
    tmask2 = ~np.isnan(calibrated)
    ras = ras[tmask2]
    decs = decs[tmask2]
    calibrated = calibrated[tmask2] 
    
    c = coordinates.SkyCoord(frame='fk5', ra=ras*u.degree, dec=decs*u.degree)
    decs = c.galactic.b.deg
    ras = c.galactic.l.deg
    
    if bank == 'A':
        cfreq = 4.575
    elif bank == 'B':
        cfreq = 5.625
    elif bank == 'C':
        cfreq = 6.125
    elif bank == 'D':
        cfreq = 7.175
    beam = 12.6 / cfreq
    beamarea = np.pi / (4. * np.log(2)) * beam**2
    pixelarea = pixbeam**2
    units = pixelarea / beamarea
    
    nrapix = int((ras.max() - ras.min()) / (pixbeam / 60.))
    ndecpix = int((decs.max() - decs.min()) / (pixbeam / 60.))
    
    datamap, hits, rabins, decbins = naive_map(calibrated, ras, decs, nrapix, ndecpix) 
    mask = hits == 0
    signal = np.zeros_like(datamap)
    signal[~mask] = datamap[~mask] / hits[~mask] * units
    
    
    radius = np.sqrt( (ras-rc)**2 + (decs-dc)**2)
    rmask = radius <= rad
    insidedata = np.zeros_like(calibrated)
    insidedata[rmask] = 10.
    innermap, innerhits, rabins, decbins = naive_map(insidedata, ras, decs, nrapix, ndecpix) 
    innerregion = innermap > 0
    
    arc = 107.2
    adc = 5.2
    radius = np.sqrt( (ras-arc)**2 + (decs-adc)**2)
    annulus = (radius >= 1. ) & (radius <= 2.)
    outerdata = np.zeros_like(calibrated)
    outerdata[annulus] = 10.
    outermap, outerhits, rabins, decbins = naive_map(outerdata, ras, decs, nrapix, ndecpix)
    outerregion = outermap > 0

    if smth is None:
        smth = beam
    if smth > 0:
        sigma = smth / pixbeam / (2. * np.sqrt(2. * np.log(2)))
        signal = gaussian_filter(signal, sigma)
        shits = gaussian_filter(hits, sigma)
        mask = shits < 1
    
    rabinsc = (rabins[1:] + rabins[:-1]) / 2.
    decbinsc = (decbins[1:] + decbins[:-1]) / 2.
    if removeplane:
        X, Y = np.meshgrid(rabinsc, decbinsc)
        XX = X.flatten()
        YY = Y.flatten()
        Z = signal.flatten()
        masks = ~np.isnan(Z) * ~innerregion.flatten()
        data = np.c_[XX[masks], YY[masks], Z[masks]]
        data2 = np.c_[XX, YY, Z]
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = linalg.lstsq(A, data[:, 2])
        res = C[0] * X + C[1] * Y + C[2]
        signal -= res
        
    signal -= np.nanmedian(signal[outerregion])
    signal[mask] = np.nan
    return np.nansum(signal[innerregion])


def aperture_photometry_cgps(rc=107.2, dc=5.2, rad=1.):
    z = np.load('../externaldata/cgps.npz')
    dlons = z['lons']
    dlats = z['lats']
    dsig = z['signal']

    data = dsig.flatten()
    area = (((dlons.max() - dlons.min())/256)**2 + ((dlats.max() - dlats.min())/256)**2) * d2r * d2r
    nu = 408.e6
    datajy = data * 2. * kboltz * (nu / clight)**2 * area * 1.e26

    lonc = 107.2
    latc = 5.2
    lonsq = (dlons - lonc)**2
    latsq = (dlats - latc)**2
    biglon = np.ones((256, 256)) * lonsq
    biglat = np.transpose(np.ones((256, 256)) * latsq)
    radius = np.sqrt(biglon + biglat).flatten()
    annulus = (radius > 1.) * (radius < 2.)
    adata = np.median(datajy[annulus])


    lonsq = (dlons - rc)**2
    latsq = (dlats - dc)**2
    biglon = np.ones((256, 256)) * lonsq
    biglat = np.transpose(np.ones((256, 256)) * latsq)
    radius = np.sqrt(biglon + biglat).flatten()
    rmask = radius <= rad
    rdata = datajy[rmask]
    return np.sum(rdata - adata)


def aperture_photometry_planck(freq, rc=107.2, dc=5.2, rad=1.):
    if freq == '30':
        planck = '../externaldata/LFI_SkyMap_030-field-IQU_1024_R2.01_full.fits'
        nu = 28.5e9
    if freq == '40':
        planck = '../externaldata/LFI_SkyMap_044-field-IQU_1024_R2.01_full.fits'
        nu = 44.1e9
    if freq == '70':
        planck = '../externaldata/LFI_SkyMap_070-field-IQU_1024_R2.01_full.fits' 
        nu = 70.3e9
    if freq == '143':
        planck = '../externaldata/HFI_SkyMap_143-field-IQU_2048_R2.02_full.fits' 
        nu = 143.e9
    if freq == '217':
        planck = '../externaldata/HFI_SkyMap_217-field-IQU_2048_R2.02_full.fits'  
        nu = 217.e9
    if freq == '353':
        planck = '../externaldata/HFI_SkyMap_353-field-IQU_2048_R2.02_full.fits' 
        nu = 353.e9
    if freq == '545':
        planck = '../externaldata/HFI_SkyMap_545-field-Int_2048_R2.02_full.fits' 
        nu = 545.e9
    if freq == '857':
        planck = '../externaldata/HFI_SkyMap_857-field-Int_2048_R2.02_full.fits' 
        nu = 857.e9

    planckmap = hp.read_map(planck, verbose=False)
    x = np.copy(planckmap)
    nside = hp.get_nside(x)
    
    vecc = hp.rotator.dir2vec(rc, dc, lonlat=True)
    rmask = hp.query_disc(nside, vecc, rad *d2r)
    
    lonc = 107.2
    latc = 5.2
    amaskout = set(hp.query_disc(nside, vecc, 2.*d2r))
    amaskin = set(hp.query_disc(nside, vecc, 1.*d2r))
    amask = np.array(list(amaskout.difference(amaskin)))
    X = hplanck * nu / (kboltz * TCMB) 
    kthermo_to_intensity = 2. * kboltz * (nu / clight)**2 * (X**2 * np.exp(X)) / (np.exp(X) - 1.)**2
    if nu < 400e9:
        y = x * kthermo_to_intensity * hp.nside2pixarea(nside) * 1.e26 
    else:
        y = x * 1e6 * hp.nside2pixarea(nside)
    rdata = y[rmask]
    adata = y[amask]
    return np.sum(rdata - np.median(adata))



