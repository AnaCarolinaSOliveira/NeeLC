import numpy as np
import healpy as hp

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
