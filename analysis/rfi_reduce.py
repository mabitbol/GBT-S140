import numpy as np
from astropy.io import fits
import matplotlib as mpl
mpl.use('agg')
import pylab as pl


def mad(x, axis=None):
    return np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)

def moving_median(x, N):
    idx = np.arange(N) + np.arange(len(x) - N + 1)[:, None]
    return np.nanmedian(x[idx], axis=1)

def reduce_scan(filename, nchannels=512, plot=True):
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

    noiser = xxon - xxoff
    mask = np.ones(N, dtype=bool)
    mask[(index % 512) == 0] = False
    normed_data = (np.nanmedian(xxoff, 0) / np.nanmedian(noiser, 0))
    normed_data[~mask] = np.nan
    mvmedspec = moving_median(normed_data, 5)
    flatdata = (normed_data[2:-2] - mvmedspec)
    mask[if_freqs < 150.e6] = False
    mask[if_freqs > 1400.e6] = False
    flatdata[~mask[2:-2]] = np.nan
    madspec = mad(flatdata)
    mask[2:-2][flatdata > (16 * madspec)] = False

    if plot:
        pl.figure()
        pl.plot(freqs[2:-2], flatdata, 'r')
        flatdata[~mask[2:-2]] = np.nan
        pl.plot(freqs[2:-2], flatdata)
        nstd = np.nanstd(flatdata)
        pl.axhline(16 * madspec, color='g')
        pl.ylim(-10*nstd, 10*nstd)
        plotname = filename.split('.fits')[0] + '_rfi.png' 
        pl.savefig(plotname)
        pl.close()

    bigmask = np.broadcast_to(mask, noiser.shape)
        
    xxoff[~bigmask] = np.nan
    xxon[~bigmask] = np.nan
    yyoff[~bigmask] = np.nan
    yyon[~bigmask] = np.nan
    xyoff[~bigmask] = np.nan
    xyon[~bigmask] = np.nan
    yxoff[~bigmask] = np.nan
    yxon[~bigmask] = np.nan

    result['xxoff'] = np.nanmean(xxoff.reshape(-1, nchannels, N//nchannels), 2)
    result['xxon'] = np.nanmean(xxon.reshape(-1, nchannels, N//nchannels), 2)
    result['yyoff'] = np.nanmean(yyoff.reshape(-1, nchannels, N//nchannels), 2)
    result['yyon'] = np.nanmean(yyon.reshape(-1, nchannels, N//nchannels), 2)
    result['xyoff'] = np.nanmean(xyoff.reshape(-1, nchannels, N//nchannels), 2)
    result['xyon'] = np.nanmean(xyon.reshape(-1, nchannels, N//nchannels), 2)
    result['yxoff'] = np.nanmean(yxoff.reshape(-1, nchannels, N//nchannels), 2)
    result['yxon'] = np.nanmean(yxon.reshape(-1, nchannels, N//nchannels), 2)

    freqs[~mask] = np.nan
    result['freqs'] = np.nanmean(freqs.reshape(nchannels, -1), 1)
    result['weight'] = np.sum(mask.reshape(nchannels, -1).astype(int), 1)
    result['valid'] = result['weight'] > 0 
    result['mask'] = mask

    for name in hdu.columns.names:
        if name not in ['DATA', 'CRVAL2','CRVAL3','TCAL']:
            if len(np.unique(hdu.data[name]))==1:
                result[name] = hdu.data[name][:1]
            else:
                result[name] = hdu.data[name]
    outfile = filename + '.rfireduced.npz'
    print "saving to",outfile
    np.savez(outfile,**result)

    return 


if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    try:
        reduce_scan(filename)
    except Exception as e:
        print "Failed on",filename,e
