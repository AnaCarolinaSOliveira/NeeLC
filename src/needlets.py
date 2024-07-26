import numpy as np
import healpy as hp

from utils import calculate_nside, arcmin2rad

def gaussian_window_function(l, theta_fwhm_arcmin):
    """
    Compute the value of the Gaussian window function for a given multipole l and FWHM theta_fwhm.

    Parameters:
    l (float): The multipole value.
    theta_fwhm (float): The full width at half maximum (FWHM) of the Gaussian window function.

    Returns:
    float: The value of the Gaussian window function for the given l and theta_fwhm.
    """
    theta_fwhm = arcmin2rad(theta_fwhm_arcmin)
    return np.nan_to_num(np.exp(-l * (l + 1) * (theta_fwhm ** 2) / (16 * np.log(2))))

def cosine_window_function(l, l_peak, l_min, l_max):
    """
    Calculate the cosine window function for a given range of values.

    Parameters:
    l (float or array-like): The input value(s) at which to calculate the window function.
    l_peak (float): The peak value of the window function.
    l_min (float): The minimum value of the window function.
    l_max (float): The maximum value of the window function.

    Returns:
    float or array-like: The calculated window function value(s) for the given input(s).
    """
    # Create a vectorized conditional using numpy's where
    condition1 = (l_min <= l) & (l < l_peak)
    condition2 = (l == l_peak)
    condition3 = (l_peak < l) & (l <= l_max)

    # Calculate the window function for each condition
    window = np.where(
        condition1, 
        np.cos(((l_peak - l) / (l_peak - l_min)) * (np.pi / 2)),
        np.where(
            condition2, 
            1,
            np.where(
                condition3, 
                np.cos(((l - l_peak) / (l_max - l_peak)) * (np.pi / 2)),
                0
            )
        )
    )
    
    return window


class Needlet(object):
    """
    Needlet class to perform needlet analysis.
    """
    def __init__(self, nbands, l_max, tol=1e-6):
        """
        Initialize the Needlet class.
        
        Parameters:
        nbands (int): The number of needlet bands.
        l_max (int): The maximum multipole value.
        tol (float): The tolerance for the filter normalization. Default is 1e-6.
        """
        self.nbands = nbands # number of bands
        self.l_max = l_max # maximum multipole
        self.l_maxs = np.ones(nbands, dtype=int) * l_max # maximum multipole for each band
        self.l_range = np.arange(self.l_max+1) # multipole range
        self.nside = np.ones(nbands, dtype=int) * calculate_nside(self.l_max) # nside for each band
        self.h_ell = np.ones((self.nbands, self.l_max+1)) # filter function for each band
        self.tol = tol # tolerance for the filter normalization

    def check_filter_normalization(self):
        """
        Check if the filter normalization is correct.

        This method checks if the sum of squares of each row of `self.h_ell` is close to 1 for all multipoles.

        Raises:
            AssertionError: If the filter normalization is not correct.
        """
        assert( np.isclose((self.h_ell**2).sum(0), 1, atol=self.tol).all() )

    def plot_filter(self):
        """
        Plot the filter functions for each band.
        """
        import matplotlib.pyplot as plt
        for j in range(self.nbands):
            plt.plot(self.l_range, self.h_ell[j], label='j=%d'%j)
        plt.xlabel(r'$\ell$', fontsize=14)
        plt.ylabel(r'$h_{\ell}^{j}$', fontsize=14)
        plt.grid(alpha=0.3)
        plt.xscale('log')
        plt.legend()
        # plt.show()


    def alm2betajk(self, alm, j):
        """
        Transform alm coefficients to needlet coefficients in the j-th band (i.e. a healpix map).

        Parameters:
        alm (array): The alm coefficients.
        j (int): The needlet band index.

        Returns:
        array: The needlet coefficients
        """
        # alm = np.atleast_2d(alm)
        # for i in range(alm.shape[0]):
        #     alm[i] = hp.almxfl(alm[i], self.h_ell[j,:])
        # print(alm.shape)
        alm = hp.almxfl(alm, self.h_ell[j,:])
        betajk = hp.alm2map(alm, lmax=self.l_maxs[j], nside=self.nside[j], verbose=False)
        return betajk

    def map2betajk(self, m, j, iter=3, use_pixel_weights=True):
        """
        Transform a map to needlet coefficients in the j-th band  (i.e. a healpix map).

        Parameters:
        m (array): The input map.      
        j (int): The needlet band index.
        iter (int): The number of iterations for the map2alm transformation. Default is 3.
        use_pixel_weights (bool): Whether to use pixel weights in the map2alm transformation. Default is True.

        Returns:
        array: The needlet coefficients.
        """
        alm = hp.map2alm(m, lmax=self.l_maxs[j], iter=iter, use_pixel_weights=use_pixel_weights)
        return self.alm2betajk(alm, j)
    


class GaussianNeedlet(Needlet):
    """
    Needlets with Gaussian window functions.
    """
    def __init__(self, theta_fwhms_arcmin, l_max, l_min=None):
        # make sure theta_fwhms_arcmin is an array of numbers in descending order
        assert(np.all(np.diff(theta_fwhms_arcmin) < 0))

        super(GaussianNeedlet, self).__init__(len(theta_fwhms_arcmin)+1, l_max)
        if l_min is None:
            self.l_mins = np.zeros(len(theta_fwhms_arcmin)+1, dtype=int)
        else:
            self.l_mins = np.ones(len(theta_fwhms_arcmin)+1, dtype=int)*l_min

        # create the Gaussian window functions
        self.h_ell[0] = gaussian_window_function(self.l_range, theta_fwhms_arcmin[0])
        for i in range(1, len(theta_fwhms_arcmin)):
            self.h_ell[i] = np.sqrt(gaussian_window_function(self.l_range, theta_fwhms_arcmin[i])**2 - gaussian_window_function(self.l_range, theta_fwhms_arcmin[i-1])**2)
        self.h_ell[-1] = np.sqrt(1 - gaussian_window_function(self.l_range, theta_fwhms_arcmin[-1])**2)

        self.check_filter_normalization()


class CosineNeedlet(Needlet):
    """
    Needlets with cosine window functions.
    """
    def __init__(self, l_peaks, l_mins, l_maxs):
        self.l_peaks = l_peaks
        assert(np.all(np.diff(l_peaks) > 0))

        super(CosineNeedlet, self).__init__(len(l_peaks), l_maxs[-1])
        self.l_mins = l_mins
        self.l_maxs = l_maxs
        self.nside = calculate_nside(self.l_maxs) * np.ones(self.nbands, dtype=int)

        # create the cosine window functions
        self.h_ell = np.zeros((self.nbands, self.l_maxs[-1]+1))
        for j in range(self.nbands):
            self.h_ell[j] = cosine_window_function(self.l_range, self.l_peaks[j], self.l_mins[j], self.l_maxs[j])

        self.check_filter_normalization()        
        