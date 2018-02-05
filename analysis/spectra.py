import numpy as np
import healpy as hp
from scipy.ndimage.filters import gaussian_filter

kboltz=1.3806503e-23 
clight=299792458.0 
hplanck=6.626068e-34 
TCMB = 2.72548 
d2r = np.pi / 180.
pi = np.pi

## These maps are smoothed to 1 degree and in units of Jy / pixel.

class AperturePhotometry:
    def __init__(self, lonc=107.2, latc=5.2, inside=80, outside=100, radius=60):
        self.lonc = lonc
        self.latc = latc
        self.inside = inside
        self.outside = outside
        self.radius = radius
        self.nside = 512

        self.load_experiments()
        self.get_aperture()
        return

    def run_experiment(self, filelist):
        fluxs = []
        errors = []
        for fl in filelist:
            xmap = hp.read_map(fl, verbose=False)
            flux, rms = self.get_flux(xmap) 
            fluxs.append(flux)
            errors.append(rms)
        return fluxs, errors

    def get_flux(self, xmap):
        rdata = xmap[self.rmask]
        adata = xmap[self.amask]
        flux = np.sum(rdata - np.median(adata))
        rms = np.std(adata) * np.sqrt(len(rdata) + pi/2. * float(len(rdata)**2) / len(adata))
        return flux, rms

    def get_aperture(self):
        vecc = hp.rotator.dir2vec(self.lonc, self.latc, lonlat=True)
        rmask = hp.query_disc(self.nside, vecc, (self.radius/60.)*d2r, inclusive=False)
        amaskout = set(hp.query_disc(self.nside, vecc, (self.outside/60.)*d2r, inclusive=False))
        amaskin = set(hp.query_disc(self.nside, vecc, (self.inside/60.)*d2r, inclusive=False))
        amask = np.array(list(amaskout.difference(amaskin)))
        self.rmask = rmask
        self.amask = amask
        return

    def get_haslam_flux(self):
        haslam0408 = hp.read_map('../externaldata/haslam408_dsds_Remazeilles2014.fits', verbose=False)
        haslam0408 = hp.ud_grade(hp.smoothing(haslam0408, fwhm=np.sqrt(60.**2 - 56.**2)/60.*d2r), self.nside)

        nu = 408.e6
        kthermo_to_intensity = 2. * kboltz * (nu / clight)**2 * hp.nside2pixarea(self.nside) * 1.e26
        y = haslam0408 * kthermo_to_intensity
        rdata = y[self.rmask]
        adata = y[self.amask]
        flux = np.sum(rdata - np.median(adata))
        rms = np.std(adata) * np.sqrt(len(rdata) + pi/2. * float(len(rdata)**2) / len(adata))
        additional_err = 2. * kboltz * (nu / clight)**2 * 1.e26 * 3. * (self.radius/60.*d2r)**2        
        return flux, np.sqrt(additional_err**2 + rms**2) 

    def get_cgps_flux(self, sbeam=60.):
        z = np.load('../externaldata/cgps.npz')
        dlons = z['lons']
        dlats = z['lats']
        dsig = z['signal']

        sigma1 = np.sqrt(sbeam**2 - 3.2**2) / 0.7471 / (2. * np.sqrt(2. * np.log(2)))
        sigma2 = np.sqrt(sbeam**2 - 2.8**2)  / 0.9961 / (2. * np.sqrt(2. * np.log(2)))
        smthsig = gaussian_filter(dsig, [sigma1, sigma2])

        area = (((dlons.max() - dlons.min())/256.) * ((dlats.max() - dlats.min())/256.)) * d2r * d2r
        nu = 408.e6
        datajy = smthsig * 2. * kboltz * (nu / clight)**2 * area * 1.e26

        lonsq = (dlons - self.lonc)**2
        latsq = (dlats - self.latc)**2
        biglon = np.ones((256, 256)) * lonsq
        biglat = np.transpose(np.ones((256, 256)) * latsq)
        radius = np.sqrt(biglon + biglat).flatten()
        annulus = (radius > (self.inside/60.)) * (radius < (self.outside/60.))

        dataflat = datajy.flatten()
        adata = dataflat[annulus]
        rmask = radius <= (self.radius / 60.)
        rdata = dataflat[rmask]
        flux = np.sum(rdata - np.median(adata))
        rms = np.std(adata) * np.sqrt(len(rdata) + pi/2. * float(len(rdata)**2) / len(adata))
        additional_err = 2. * kboltz * (nu / clight)**2 * 1.e26 * 3. * (self.radius/60.*d2r)**2        
        return flux, np.sqrt(additional_err**2 + rms**2) 

    def get_21cm_flux(self, sbeam=60.):
        stockert = hp.read_map('../externaldata/STOCKERT+VILLA-ELISA_1420MHz_1_256.fits', verbose=False) * 1.e-3
        newbeam = np.sqrt(sbeam**2 - 36.**2)/60.
        stockert = hp.ud_grade(hp.smoothing(stockert, fwhm=newbeam*d2r), self.nside)

        nu = 1420.e6
        kthermo_to_intensity = 2. * kboltz * (nu / clight)**2 * hp.nside2pixarea(self.nside) * 1.e26
        y = stockert * kthermo_to_intensity
        rdata = y[self.rmask]
        adata = y[self.amask]
        flux = np.sum(rdata - np.median(adata)) * 1.55
        rms = np.std(adata) * np.sqrt(len(rdata) + pi/2. * float(len(rdata)**2) / len(adata))
        return flux, np.sqrt(rms**2 + (0.1 * flux)**2)


    def load_experiments(self):
        self.wmapfiles = ['../externaldata/wmap_smoothed_22.fits', \
                         '../externaldata/wmap_smoothed_32.fits',\
                         '../externaldata/wmap_smoothed_40.fits',\
                         '../externaldata/wmap_smoothed_60.fits',\
                         '../externaldata/wmap_smoothed_93.fits']
        self.wmapfreqs = np.array([22.71, 32.95, 40.65, 60.64, 93.44]) * 1e9

        self.planckfiles = ['../externaldata/planck_smoothed_28.fits', \
                           '../externaldata/planck_smoothed_44.fits', \
                           '../externaldata/planck_smoothed_70.fits', \
                           '../externaldata/planck_smoothed_143.fits', \
                           '../externaldata/planck_smoothed_217.fits', \
                           '../externaldata/planck_smoothed_353.fits', \
                           '../externaldata/planck_smoothed_545.fits', \
                           '../externaldata/planck_smoothed_857.fits']
        self.planckfreqs = np.array([28.4, 44.1, 70.4, 143., 217., 353., 545., 857.]) * 1e9
        self.planckbeams = np.array([32.3, 27.1, 13.3, 7.3, 5., 4.8, 4.7, 4.3])

        self.irisfiles = ['../externaldata/iris_smoothed_25000.fits', \
                         '../externaldata/iris_smoothed_12000.fits', \
                         '../externaldata/iris_smoothed_5000.fits', \
                         '../externaldata/iris_smoothed_3000.fits']
        self.irisfreqs = np.array([25000, 12000, 5000, 3000]) * 1.e9
        self.irisbeams = np.array([3.8, 3.8, 4.0, 4.3])

        self.dirbefiles = ['../externaldata/dirbe_smoothed_240.fits', \
                          '../externaldata/dirbe_smoothed_136.fits', \
                          '../externaldata/dirbe_smoothed_85.fits', \
                          '../externaldata/dirbe_smoothed_61.fits', \
                          '../externaldata/dirbe_smoothed_25.fits', \
                          '../externaldata/dirbe_smoothed_12.fits', \
                          '../externaldata/dirbe_smoothed_5.fits', \
                          '../externaldata/dirbe_smoothed_3.fits', \
                          '../externaldata/dirbe_smoothed_2.fits', \
                          '../externaldata/dirbe_smoothed_1.fits']
        self.dirbefreqs = np.array([240, 136.36, 85.71, 61.22, 25, 12, 5, 3, 2.1428, 1.25]) * 1.e12
        self.dirbebeams = np.array([37.6, 41.0, 39.0, 41.5, 41.0, 41.5, 42.3, 41.0, 40.4, 39.5])
        return


