import numpy as np
import glob

class GBT:
    def __init__():
        # hardcoding some settings. this are in the fits and npz files but let's put them here for easy loading
        self.tint = 0.04    #seconds
        self.swper = 0.04   #seconds
        self.dt = 0.01957342
        self.freqs_A = self.calc_freqs(4575.e6)
        self.freqs_B = self.calc_freqs(5625.e6)
        self.freqs_C = self.calc_freqs(6125.e6)
        self.freqs_D = self.calc_freqs(7175.e6)
        
    def calc_freqs(center_freq):
        N = 2**14
        df = 1500. / N * 1.e6
        centerbin = N / 2
        index = np.arange(N)
        if_freqs = index * df
        freqs = (center_freq - df * (index - centerbin))[::-1]
        return freqs
        
    def get_fnames(bank, source, scantype='', session='', tail='rfireduced.npz'):
        datadir = '/data2/GBT/' 
        datadir += source + '/'
        if scantype == 'OnOff' or scantype == 'Spider' or scantype == 'Daisy':
            datadir += scantype +'/'
        fnames = glob.glob(datadir+'
