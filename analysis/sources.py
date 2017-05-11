import numpy as np

# Absolutely calibrated flux densities from Perley and Butler 2012
# http://iopscience.iop.org/article/10.1088/0067-0049/204/2/19/pdf
# need to propagate errors
# need to get fluxes for 3C245, 3C273, and 3C280


def S_3C295(freqs):
    a0 = 1.4866 
    a1 = -0.7871
    a2 = -0.3440
    a3 = 0.0749
    f = freqs * 1.e-9
    logf = np.log10(f)
    logS = a0 + a1*logf + a2*logf**2 + a3*logf**3
    return 10.**(logS)

def S_3C286(freqs):
    a0 = 1.2515 
    a1 = -0.4605
    a2 = -0.1715
    a3 = 0.0336
    f = freqs * 1.e-9
    logf = np.log10(f)
    logS = a0 + a1*logf + a2*logf**2 + a3*logf**3
    return 10.**(logS)



