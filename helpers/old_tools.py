import numpy.fft as npfft
import numpy as np

def calc_ispec(kk, ll, wv, _var_dens, averaging = True, truncate=True, nd_wavenumber=False, nfactor = 1):
    """Compute isotropic spectrum `phr` from 2D spectrum of variable signal2d.
    Parameters
    ----------
    var_dens : squared modulus of fourier coefficients like this:
        np.abs(signal2d_fft)**2/m.M**2
    averaging: If True, spectral density is estimated with averaging over circles,
        otherwise summation is used and Parseval identity holds
    truncate: If True, maximum wavenumber corresponds to inner circle in Fourier space,
        otherwise - outer circle
    
    nd_wavenumber: If True, wavenumber is nondimensional: 
        minimum wavenumber is 1 and corresponds to domain length/width,
        otherwise - wavenumber is dimensional [m^-1]
    nfactor: width of the bin in sqrt(dk^2+dl^2) units
    Returns
    -------
    kr : array
        isotropic wavenumber
    phr : array
        isotropic spectrum
    Normalization:
    signal2d.var()/2 = phr.sum() * (kr[1] - kr[0])
    """

    # account for complex conjugate
    var_dens = np.copy(_var_dens)
    var_dens[...,0] /= 2
    var_dens[...,-1] /= 2

    ll_max = np.abs(ll).max()
    kk_max = np.abs(kk).max()

    dk = kk[1] - kk[0]
    dl = ll[1] - ll[0]

    if truncate:
        kmax = np.minimum(ll_max, kk_max)
    else:
        kmax = np.sqrt(ll_max**2 + kk_max**2)
    
    kmin = np.minimum(dk, dl)

    dkr = np.sqrt(dk**2 + dl**2) * nfactor

    # left border of bins
    kr = np.arange(kmin, kmax-dkr, dkr)
    
    phr = np.zeros(kr.size)

    for i in range(kr.size):
        if averaging:
            fkr =  (wv>=kr[i]) & (wv<=kr[i]+dkr)    
            if fkr.sum() == 0:
                phr[i] = 0.
            else:
                phr[i] = var_dens[fkr].mean() * (kr[i]+dkr/2) * np.pi / (dk * dl)
        else:
            fkr =  (wv>=kr[i]) & (wv<kr[i]+dkr)
            phr[i] = var_dens[fkr].sum() / dkr
    
    # convert left border of the bin to center
    kr = kr + dkr/2

    # convert to non-dimensional wavenumber 
    # preserving integral over spectrum
    if nd_wavenumber:
        kr = kr / kmin
        phr = phr * kmin

    return kr, phr

def compute_spectrum(_u, window, dx=1., dy=1., **kw):
    '''
    Input: np.array of size Ntimes * Nz * Ny * Nx,
    average dx,dy over domain (in meters;
    for spherical geometry average is given over center cells)
    '''

    # If NaNs (occasionally) are present (some boundary is defined as such,
    # but not necessary it is defined with NaNs), they are changed to 0
    u = np.nan_to_num(_u)

    nx = u.shape[-1]
    ny = u.shape[-2]

    # subtract spatial mean (as our spectra ignore this characteristic,
    # and there may be spectral leakage in a case of window)
    u = u - u.mean(axis=(-1,-2), keepdims=True)

    # apply window
    if window == 'rect':
        Wnd = np.ones((ny,nx))
    elif window == 'hanning':
        Wnd = np.outer(np.hanning(ny),np.hanning(nx))
    elif window == 'hamming':
        Wnd = np.outer(np.hamming(ny),np.hamming(nx))
    elif window == 'bartlett':
        Wnd = np.outer(np.bartlett(ny),np.bartlett(nx))
    elif window == 'blackman':
        Wnd = np.outer(np.blackman(ny),np.blackman(nx))
    elif window == 'kaiser':
        Wnd = np.outer(np.kaiser(ny,14),np.kaiser(nx,14))
    else:
        print('wrong window')

    # compensation of Parseval identity
    Wnd_sqr = (Wnd**2).mean()

    # Pointwise multiplication occurs only at last two dimensions
    # see https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
    u = u * Wnd

    uf = npfft.rfftn(u, axes=(-2,-1))
    M = u.shape[-1] * u.shape[-2] # total number of points

    u2 = (np.abs(uf)**2 / M**2) / Wnd_sqr

    if len(u2.shape) == 3:
        u2 = u2.mean(axis=0)
    elif len(u2.shape) > 3:
        print('error')

    # maximum wavenumber is 1/(2*d) = pi/dx = pi/dy
    kx = npfft.rfftfreq(nx,d=dx/(2*np.pi))
    ky = npfft.fftfreq(ny,d=dy/(2*np.pi))

    Kx, Ky = np.meshgrid(kx, ky)
    K = np.sqrt(Kx**2+Ky**2)
    
    return calc_ispec(kx, ky, K, u2, **kw)