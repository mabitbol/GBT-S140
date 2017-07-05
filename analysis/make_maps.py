import numpy as np

def naive_map(data,ra,dec,ra_num_bins=60,dec_num_bins=50):
    ra_bins = np.linspace(ra.min(),ra.max(),ra_num_bins)
    dec_bins = np.linspace(dec.min(),dec.max(),dec_num_bins)
    p,_,_ = np.histogram2d(ra,dec,bins=(ra_bins,dec_bins),weights=data)
    hits,_,_ = np.histogram2d(ra,dec,bins=(ra_bins,dec_bins))
    return p, hits, ra_bins, dec_bins

def get_data(bank='A', session=5):
    fnames = glob.glob('/data2/GBT/S140/Daisy/*/*_0'+str(session)+'_*'+bank+'*.rfireduced_new.npz')
    fnames.sort()
    calibrations = np.load('/home/mabitbol/GBT-S140/calibrations/calibration_3C295_0'+str(session)+'_'+bank+'.npy').item()
    fdiode = calibrations['freqs']
    pdiode = calibrations['pdiode']
    #pdiode1 = calibrations['pdiode1']
    #meanpower = 0.5 * (pdiode + pdiode1)
    calibrated = [] 
    ras = []
    decs = []
    meansub = []
    azs = []
    els = []
    gain0 = []
    gain1 = []
    for fname in fnames:
        nz = np.load(fname)
        ra = nz['sra']
        dec = nz['sdec']
        mask = nz['valid']
        az = nz['AZIMUTH'][::8]
        el = nz['ELEVATIO'][::8]
        freqs = nz['freqs']
            
        gain = pdiode / ( nz['xxon'] - nz['xxoff'])
        calibd = nz['xxoff'] * gain
        
        mask[:200] = False
        mask[400:] = False
    
        calibd = calibd[:, mask]

        calibrated = np.concatenate([calibrated, np.nanmean(calibd - np.nanmedian(calibd,0), 1)])
        meansub = np.concatenate([meansub, np.nanmean(calibd,1) - np.nanmedian(np.nanmean(calibd,1))])
        ras = np.concatenate([ras, ra])
        decs = np.concatenate([decs, dec])
        azs = np.concatenate([azs, az])
        els = np.concatenate([els, el])

        gain0 = np.concatenate([gain0, np.nanmean(gain[:, mask], 0)])
        gain1 = np.concatenate([gain1, np.nanmean(gain[:, mask], 1)])
    return calibrated, meansub, ras, decs, azs, els, gain0, gain1


def get_maps(bank='A', session=5, doplot=True, dosave=True):
    calibrated, meansub, ras, decs, azs, els, gain0, gain1 = get_data(bank, session)
    
    tmask = np.ones(len(calibrated), dtype=bool)
    ts = np.arange(len(calibrated))
    if session == 5:
        tmask = (ts < 42000) | (ts > 50000 )
    if session == 3:
        if bank == 'A':
            tmask[(ts > 35000) * (ts < 50000 )] = False
            tmask[(ts > 162500)] = False
        if bank == 'B':
            tmask[(ts > 62000) * (ts < 72000 )] = False
            tmask[(ts > 120000) * (ts < 128000 )] = False
        else:
            tmask[(ts > 38000) * (ts < 39000 )] = False
        
    if bank == 'A':
        cfreq = 4.575
    elif bank == 'B':
        cfreq = 5.625
    elif bank == 'C':
        cfreq = 6.125
    elif bank == 'D':
        cfreq = 7.175
    beam = 12.6 / cfreq
    rapix = int((ras.max() - ras.min()) / (beam / 60.))
    decpix = int((decs.max() - decs.min()) / (beam / 60.))
    
    datamap, hits, rabins, decbins = naive_map(calibrated[tmask], ras[tmask], decs[tmask], rapix, decpix)
    mask = hits == 0
    signal = datamap / hits
    signal[mask] = np.nan
    
    varmap = np.nanstd(calibrated[tmask])/np.sqrt(hits)
    mask = hits == 0
    varmap[mask] = np.nan
    
    if doplot:
        figure()
        plot(calibrated[tmask])
        ylim(-1, 1)

        figure()
        pc = pcolormesh(rabins, decbins, signal.T)
        clim(-0.1, 0.1)
        cb = colorbar()
        xlabel('RA [degrees]')
        ylabel('DEC [degrees]')
        cb.set_label('Flux [Jy]')

        figure()
        pc = pcolormesh(rabins, decbins, hits.T)
        clim(1, 40)
        cb = colorbar()
        xlabel('RA [degrees]')
        ylabel('DEC [degrees]')
        cb.set_label('Hits')

        figure()
        pc = pcolormesh(rabins, decbins, varmap.T)
        clim(0, 0.02)
        cb = colorbar()
        xlabel('RA [degrees]')
        ylabel('DEC [degrees]')
        cb.set_label('Sqrt Variance [Jy]')
    
    if dosave:
        z = {'rabins': rabins, 'decbins':decbins, 'signal':signal.T, 'hits':hits.T, 'weight':varmap.T, 
            'calibrated':calibrated, 'meansub':meansub, 'ras':ras, 'decs':decs, 'azs':azs, 'els':els, 
            'gain0':gain0, 'gain1':gain1, 'tmask':tmask}
        np.savez('/home/mabitbol/GBT-S140/datamaps/datamaps_'+bank+'_'+str(session), **z)
    return calibrated, gain0, gain1


def calculate_flux(bank='A', session='5', doplot=True):
    dataf = np.load('/home/mabitbol/GBT-S140/datamaps/datamaps_'+bank+'_'+session+'.npz')
    tmask = dataf['tmask']
    ras = dataf['ras'][tmask]
    decs = dataf['decs'][tmask]
    calibrated = dataf['calibrated'][tmask]
    
    radius = np.sqrt( (ras-rc)**2 + (decs-dc)**2)
    rmask = radius < 1.
    datamap, hits, rabins, decbins = naive_map(calibrated[rmask], ras[rmask], decs[rmask], 120, 120)
    mask = hits == 0
    signal = datamap / hits
    signal[mask] = np.nan
    
    if doplot:
        figure()
        pc = pcolormesh(rabins, decbins, signal.T)
        clim(-0.1, 0.1)
        cb = colorbar()
        xlabel('RA [degrees]')
        ylabel('DEC [degrees]')
        cb.set_label('Flux [Jy]')
    
    annulus = (radius > 80./60.) & (radius < 2.)
    datamap, hits, rabins, decbins = naive_map(calibrated[annulus], ras[annulus], decs[annulus], 240, 240)
    mask = hits == 0
    nullmap = datamap / hits
    nullmap[mask] = np.nan
    
    if doplot:
        figure()
        pc = pcolormesh(rabins, decbins, nullmap.T)
        clim(-0.1, 0.1)
        cb = colorbar()
        xlabel('RA [degrees]')
        ylabel('DEC [degrees]')
        cb.set_label('Flux [Jy]')
    
    beam1arcmin = ( (1./60.) * (np.pi / 180.) )**2
    if bank == 'A':
        cfreq = 4.575
    elif bank == 'B':
        cfreq = 5.625
    elif bank == 'C':
        cfreq = 6.125
    elif bank == 'D':
        cfreq = 7.175
    beam = 12.6 / cfreq
    actualbeam = ( (beam/60.) * (np.pi / 180.) )**2
    print ( np.nansum(signal - np.nanmedian(nullmap)) ) * beam1arcmin / actualbeam, 'Jy'
    return ( np.nansum(signal - np.nanmedian(nullmap)) ) * beam1arcmin / actualbeam
