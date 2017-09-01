import numpy as np
from astropy.io import fits
import glob


def mad(x, axis=None):
    return np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)

def reduce_scan(filename, nchannels=512):
    f = fits.open(filename)
    hdu = f[1]

    data = hdu.data['DATA']
    result = {}

    N = 2**14
    df = hdu.data['CDELT1'][0]
    center_freq = hdu.data['CRVAL1'][0]
    centerbin = hdu.data['CRPIX1'][0]
    index = np.arange(N)
    if_freqs = freqstest = index * (-df) # in Hz
    freqs = (center_freq + df * (index + 1 - centerbin))[::-1]
    
    print hdu.data['EQUINOX'][0]
    print hdu.data['RADESYS'][0]

    if False:
        result['source'] = hdu.data['PROCSCAN'][0]
        result['centerfreq'] = center_freq
        result['ra'] = hdu.data['CRVAL2']
        result['dec'] = hdu.data['CRVAL3']
        result['sra'] = hdu.data['CRVAL2'][::8]
        result['sdec'] = hdu.data['CRVAL3'][::8]
        result['tcalx'] = hdu.data['TCAL'][:1]
        result['tcaly'] = hdu.data['TCAL'][2:3]

        xxoff = data[::8,::-1]
        xxon = data[1::8,::-1]
        yyoff = data[2::8,::-1]
        yyon = data[3::8,::-1]
        xyoff = data[4::8,::-1]
        xyon = data[5::8,::-1]
        yxoff = data[6::8,::-1]
        yxon = data[7::8,::-1]

        mask = np.ones(N, dtype=bool)
        mask[(index % 512) == 511] = False
        mask[if_freqs < 150.e6] = False
        mask[if_freqs > 1400.e6] = False

        if filename.split('.')[-2] == 'A':
            mask[(freqs >= 4.225e9) & (freqs <= 4.375e9)] = False

        xxoff[:, ~mask] = np.nan

        ratio = np.zeros(N)
        ratio[mask] = np.nanstd(xxoff[:, mask], 0) / np.nanmean(xxoff[:, mask], 0)
        ratio[~mask] = np.nan
        ratio_mad = 5. / 0.67449 * mad(ratio[mask])
        bad = np.zeros_like(mask)
        bad[mask] = np.abs(ratio[mask] - np.nanmean(ratio[mask])) > ratio_mad
        mask[bad] = False
        xxoff[:, ~mask] = np.nan

        msdata = np.zeros_like(xxoff)
        msdata[:, mask] = xxoff[:, mask] - np.nanmean(xxoff[:, mask], 0)
        msdata[:, ~mask] = np.nan

        speck = np.zeros(xxoff.shape[1])
        speck[mask] = np.nanmean((msdata[:, mask]**4), 0) / np.nanmean((msdata[:, mask]**2), 0)**2
        speck[~mask] = np.nan
        speck_mad = 5. / 0.67449 * mad(speck[mask])
        bad = np.zeros_like(mask)
        bad[mask] = np.abs(speck[mask] - np.nanmean(speck[mask])) > speck_mad
        mask[bad] = False

        xxoff[:, ~mask] = np.nan
        xxon[:, ~mask] = np.nan
        yyoff[:, ~mask] = np.nan
        yyon[:, ~mask] = np.nan
        xyoff[:, ~mask] = np.nan
        xyon[:, ~mask] = np.nan
        yxoff[:, ~mask] = np.nan
        yxon[:, ~mask] = np.nan

        result['xxoff'] = np.nanmean(xxoff.reshape(-1, nchannels, N//nchannels), 2)
        result['xxon'] = np.nanmean(xxon.reshape(-1, nchannels, N//nchannels), 2)
        result['yyoff'] = np.nanmean(yyoff.reshape(-1, nchannels, N//nchannels), 2)
        result['yyon'] = np.nanmean(yyon.reshape(-1, nchannels, N//nchannels), 2)
        result['xyoff'] = np.nanmean(xyoff.reshape(-1, nchannels, N//nchannels), 2)
        result['xyon'] = np.nanmean(xyon.reshape(-1, nchannels, N//nchannels), 2)
        result['yxoff'] = np.nanmean(yxoff.reshape(-1, nchannels, N//nchannels), 2)
        result['yxon'] = np.nanmean(yxon.reshape(-1, nchannels, N//nchannels), 2)

        result['original_freqs'] = freqs
        result['freqs'] = np.nanmean(freqs.reshape(nchannels, -1), 1)
        result['nweight'] = np.sum(mask.reshape(nchannels, -1).astype(int), 1)
        result['valid'] = result['nweight'] > 0 
        result['mask'] = mask

        freqs[~mask] = np.nan
        result['weighted_freqs'] = np.nanmean(freqs.reshape(nchannels, -1), 1)

        for name in hdu.columns.names:
            if name not in ['DATA', 'CRVAL2','CRVAL3','TCAL']:
                if len(np.unique(hdu.data[name]))==1:
                    result[name] = hdu.data[name][:1]
                else:
                    result[name] = hdu.data[name]
        #outfile = filename + '.rfireduced_new.npz'
        #print "saving to",outfile
        #np.savez(outfile,**result)
    return 


def run():
    fnames = glob.glob('/data2/GBT/*/Daisy/*/*.fits')
    fnames.sort()
    for filename in fnames:
        try:
            reduce_scan(filename)
        except Exception as e:
            print "Failed on",filename,e
            

if __name__ == "__main__":
    run()
