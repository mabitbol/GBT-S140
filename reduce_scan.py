from astropy.io import fits
import numpy as np

def reduce_scan(filename,final_channels=512):
    result = {}
    f = fits.open(filename,memmap=True)
    h = f[1]
    result['ra'] = h.data['CRVAL2']
    result['dec'] = h.data['CRVAL3']
    result['sra'] = h.data['CRVAL2'][::8]
    result['sdec'] = h.data['CRVAL3'][::8]
    result['tcalx'] = h.data['TCAL'][:1]
    result['tcaly'] = h.data['TCAL'][2:3]
    d = h.data['DATA']
    result['xxoff'] = d[::8,:].reshape((-1,final_channels,16384//final_channels)).mean(2)
    result['xxon'] = d[1::8,:].reshape((-1,final_channels,16384//final_channels)).mean(2)
    result['yyoff'] = d[2::8,:].reshape((-1,final_channels,16384//final_channels)).mean(2)
    result['yyon'] = d[3::8,:].reshape((-1,final_channels,16384//final_channels)).mean(2)
    result['xyoff'] = d[4::8,:].reshape((-1,final_channels,16384//final_channels)).mean(2)
    result['xyon'] = d[5::8,:].reshape((-1,final_channels,16384//final_channels)).mean(2)
    result['yxoff'] = d[6::8,:].reshape((-1,final_channels,16384//final_channels)).mean(2)
    result['yxon'] = d[7::8,:].reshape((-1,final_channels,16384//final_channels)).mean(2)
    for name in h.columns.names:
        if name not in ['DATA', 'CRVAL2','CRVAL3','TCAL']:
            if len(np.unique(h.data[name]))==1:
                #print "found constant column",name
                result[name] = h.data[name][:1]
            else:
                result[name] = h.data[name]
    outfile = filename + '.reduced.npz'
    print "saving to",outfile
    np.savez(outfile,**result)
    
if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    try:
        reduce_scan(filename)
    except Exception as e:
        print "Failed on",filename,e
