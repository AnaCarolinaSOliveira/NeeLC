import numpy as np
import healpy as hp

####### constants ########

T_CMB = 2.726 # K
H_PLANCK = 6.626e-34 # Js
K_B = 1.38065e-23 # J/K
T_CIB = 25 # K
C = 2.998e8 # m/s

##########################


def reduce_lmax(alm, lmax=4000, verbose=False):
    """
    (Stolen from Yuuki Omori)
    Reduce the lmax of a alm array.

    Parameters:
    alm (array): The input alm array.
    lmax (int): The desired lmax value to reduce to. Default is 4000.

    Returns:
    array: The reduced alm array.
    
    """
    lmaxin  = hp.Alm.getlmax(alm.shape[0])
    if lmaxin <= lmax:
        if verbose:
            print( "-- No need to reduce lmax: lmax_in=%g <= lmax_out=%g"%(lmaxin,lmax) )
        return alm
    elif verbose:
        print( "-- Reducing lmax: lmax_in=%g -> lmax_out=%g"%(lmaxin,lmax) )
    almout  = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex_)
    oldi = 0
    oldf = 0
    newi = 0
    newf = 0
    dl = lmaxin - lmax
    for i in range(0, lmax + 1):
        oldf = oldi + lmaxin + 1 - i
        newf = newi + lmax + 1 - i
        almout[newi:newf] = alm[oldi:oldf - dl]
        oldi = oldf
        newi = newf
    return almout


def calculate_nside(l):
    """
    Calculate the smallest power of 2 greater than l/2.

    Parameters:
    l (int or array-like): The input value or array.

    Returns:
    int or array-like: The smallest power of 2 greater than l/2.
    """
    l = np.atleast_1d(l)
    return 2 ** np.ceil(np.log2(l / 2)).astype(int)


def arcmin2rad(arcmin):
    """
    Convert arcminutes to radians.

    Parameters:
    arcmin (float or array-like): The input value or array in arcminutes.

    Returns:
    float or array-like: The value or array converted to radians.
    """
    return arcmin * np.pi / (180. * 60.)


def masked_smoothing(U, fwhm=5.0, use_pixel_weights=True):
    """
    Smooth a map with a given fwhm in degrees, masking the input map and taking care of the mask in the smoothing.
    (see https://stackoverflow.com/questions/50009141/) 

    Parameters:
    U (array): The input map.
    fwhm (float): The full width at half maximum of the smoothing kernel in degrees.

    Returns:
    array: The smoothed map.
    """ 
    V = U.copy()
    V[U != U] = 0
    VV = hp.smoothing(V, fwhm=np.radians(fwhm), use_pixel_weights=use_pixel_weights)
    W = 0 * U.copy() + 1
    W[U != U] = 0
    WW = hp.smoothing(W, fwhm=np.radians(fwhm), use_pixel_weights=use_pixel_weights)
    return VV / WW

def bin_spectrum(cl, new_edges):
    sum_c = lambda l: (2*l + 1)*cl[l]
    modes = lambda l: (2*l + 1)
    pwr_binned = np.zeros([len(new_edges)-1])
    for e in range(len(new_edges[:-1])):
        bin_range = np.arange(new_edges[e],new_edges[e+1],1)
        n_modes = 0
        c_b = 0
        for l in bin_range:
            n_modes += modes(l)
            c_b += sum_c(l)
        pwr_binned[e] = c_b / n_modes
    return pwr_binned

def rebin_array(array, newbins, oldbins):
    array_rebin = np.zeros(newbins[-1]+1)
    for i in range(len(array)):
        array_rebin[oldbins[i]:oldbins[i+1]] = array[i]
    return array_rebin

def cl2dl(Cl, Lrange):
    factor = (Lrange*(Lrange+1))/(2*np.pi)
    return Cl*factor

def f_sz(nus):
    """
    Spectral dependence of the tSZ effect, for nus in GHz.
    """
    X = (H_PLANCK * nus*1e9)/(K_B * T_CMB) 
    return X * (np.exp(X)+1.)/(np.exp(X)-1.) - 4.

def f_cib(nus, beta):
    """
    Spectral dependence of the CIB, for nus in GHz.
    """
    X = (H_PLANCK * nus*1e9)/(K_B * T_CMB)  
    F = (H_PLANCK * nus*1e9)/(K_B * T_CIB) 
    return  (nus**beta) * ((np.exp(X)-1.)**2.) / ((np.exp(F)-1.)*X*np.exp(X))

def convert_MJy_per_sr_to_muK(map_jysr, nu):
    """
    Converts map in units of MJy/sr to thermodynamic units,
    muK_CMB. Input nu must be in GHz.
    """
    def f(nu):
        x = H_PLANCK * nu / (K_B * T_CMB)
        return x**2 * np.exp(x) / np.expm1(x) ** 2

    factor = (f(nu*1e9) * 2 * K_B * (nu*1e9)**2 / C**2) * 1e14 # MJy / sr / muK
    return map_jysr / factor

class DetSpecs(object):

    def __init__(self, det='SPT'):

        self.det = det

        if det=='SPT':
            self.bands = np.array([95.,150.,220.]) # [GHz]
            self.nb = len(self.bands) # number of bands
            self.beam_size_fwhm_arcmin = np.array([1.57,1.17,1.04]) # in arcmin
            self.beam_size_fwhm = arcmin2rad(self.beam_size_fwhm_arcmin)
            self.sigma_beam = self.beam_size_fwhm / np.sqrt(8.*np.log(2.)) # fwhm to sigma

            self.white_noise_arcmin = np.array([3.0, 2.5, 8.9]) # in muK-arcmin
            self.white_noise = arcmin2rad(self.white_noise_arcmin)

            self.l_knee = np.array([1200., 2200., 2300.])
            self.alpha = np.array([-3.0, -4.0, -4.0])

        elif det=='Planck':
            self.bands = np.array([30.,44.,70.,100.,143.,217.,353.,545.,857.]) # [GHz]
            self.lfi_bands = self.bands[0:3] 
            self.hfi_bands = self.bands[3:] 
            self.nb = len(self.bands) # number of bands       

    def beam_function(self, L):
        return np.exp(-0.5 * np.outer(L**2, self.sigma_beam**2)).T
    
    def noise_eff(self, L):
        if not isinstance(L, np.ndarray):
            L = np.array(L)

        noise_eff = np.zeros([self.nb, len(L)])
        beam = self.beam_function(L)
        for b in np.arange(self.nb):
            noise_eff[b] = (np.square(self.white_noise[b])*(1+(L/self.l_knee[b])**self.alpha[b]))/np.square(beam[b]) # in muK^2-radians^2  
        noise_eff = np.where(np.isinf(noise_eff), 0, noise_eff)

        return noise_eff
