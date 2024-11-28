import numpy as np
import healpy as hp
import os, sys
from astropy.io import fits
import pickle 

####### constants ########

T_CMB = 2.7255 # K
H_PLANCK = 6.62607004e-34 # Js
K_B = 1.38064852e-23 # J/K
T_CIB = 25 # K
C = 2.99792458e8 # m/s

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
    # return 2 ** np.ceil(np.log2(l / 2)).astype(int)
    nside = 2 ** np.ceil(np.log2(l / 2)).astype(int)
    return np.maximum(nside, 32)


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


def f_sz(nus, bandpassed=False):
    """
    Spectral dependence of the tSZ effect, for nus in GHz.
    """
    if isinstance(nus, int):
        nus = np.array([nus])

    if bandpassed:
        f_sz_bp = np.zeros(len(nus))
        for idx, nu in enumerate(nus):
            if int(nu) in [95, 150, 220]:
                spt = DetSpecs(det='SPT3G_main')
                freqs, b = spt.load_bandpass(nu)
            elif nu in [100, 143, 217, 353, 545, 857]:
                planck = DetSpecs(det='Planck')
                freqs, b = planck.load_bandpass(nu)
            X = (H_PLANCK * freqs*1e9)/(K_B * T_CMB)
            fsz = X * (np.exp(X)+1.)/(np.exp(X)-1.) - 4.
            valid_mask = ~np.isnan(fsz) & ~np.isinf(fsz)            
            if np.any(valid_mask):
                f_sz_bp[idx] = np.trapz(b[valid_mask] * fsz[valid_mask], freqs[valid_mask]) / np.trapz(b[valid_mask], freqs[valid_mask])
            else:
                f_sz_bp[idx] = np.nan
        return f_sz_bp
    else:
        X = (H_PLANCK * nus*1e9)/(K_B * T_CMB) 
        return X * (np.exp(X)+1.)/(np.exp(X)-1.) - 4.
    

def f_cib(nus, beta):
    """
    Spectral dependence of the CIB, for nus in GHz.
    """
    X = (H_PLANCK * nus*1e9)/(K_B * T_CMB)  
    F = (H_PLANCK * nus*1e9)/(K_B * T_CIB) 
    return  (nus**beta) * ((np.exp(X)-1.)**2.) / ((np.exp(F)-1.)*X*np.exp(X))

def dBdT(nus):
    X = (H_PLANCK * nus*1e9)/(K_B * T_CMB)  
    factor = 2 * H_PLANCK * ((nus*1e9)**3) * X / (C**2) / T_CMB
    exp_term = (np.exp(X)) / (np.exp(X) - 1)**2
    return factor * exp_term

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


########## Transfer function ##########

def almtf2d(tf, lmax, lmin=0, mmin=0, bl=None):
    """
    From Kimmy Wu.
    
    bl: 1D B(ell), start at ell=0    
    """

    tf2d = np.zeros([lmax+1,lmax+1])
    #fill non-zero (l,m) with ones
    for l in range(0,lmax+1):
        tf2d[0:l+1 , l] = 1.0
        
    tf2d *= tf[None, :]
    
    if bl is not None:
        tf2d *= bl[None, :]

    tf2d[:mmin,:] = 0
    tf2d[:,:lmin] = 0

    return tf2d

def grid2alm(grid):
    """
    From Kimmy Wu.
    
    Convert 2d grid back to Healpix alm
    """
    lmax = grid.shape[0]-1
    alm=np.zeros(hp.Alm.getsize(lmax),dtype=np.complex_)
    for l in range(0,lmax+1):
        for m in range(0,l+1):
            # l,m
            alm[hp.Alm.getidx(lmax,l,m)]=grid[m,l]
            #alm[hp.Alm.getidx(lmax,i,np.arange(i+1)-1)]=grid[:i+1,i]
    return alm


#######################################


class DetSpecs(object):

    def __init__(self, det):
        
        allowed_dets = ['SPT3G', 'SPT3G_main', 'SPT3G_winter', 'Planck']
        assert det in allowed_dets, f"det must be one of {allowed_dets}, but got '{det}'."

        self.det = det
        
        if self.det=='SPT3G' or self.det=='SPT3G_main' or self.det=='SPT3G_winter':

            self.bands = np.array([95.,150.,220.]) # [GHz]
            self.beam_size_fwhm_arcmin = np.array([1.57,1.17,1.04]) # in arcmin

            if self.det=='SPT3G_main':

                self.white_noise_arcmin = np.array([3.0, 2.5, 8.9]) # in muK-arcmin
                self.white_noise = arcmin2rad(self.white_noise_arcmin)
                self.l_knee = np.array([1200., 2200., 2300.])
                self.alpha = np.array([-3.0, -4.0, -4.0])

            elif self.det=='SPT3G' or self.det=='SPT3G_winter':

                self.white_noise_arcmin = np.array([5.36, 4.21, 15.72]) # in muK-arcmin
                self.white_noise = arcmin2rad(self.white_noise_arcmin)
                self.l_knee = np.array([1052, 2023, 1873])
                self.alpha = np.array([-4.68, -4.11, -4.22])

        elif self.det=='Planck':
            
            self.bands = np.array([30.,44.,70.,100.,143.,217.,353.,545.,857.]) # [GHz]
            self.lfi_bands = self.bands[0:3] 
            self.hfi_bands = self.bands[3:] 
            self.beam_size_fwhm_arcmin = np.array([32.29,27.94,13.08,9.66,7.22,4.90,4.92,4.67,4.22]) # in arcmin
            
        self.nb = len(self.bands) # number of bands     
        self.beam_size_fwhm = arcmin2rad(self.beam_size_fwhm_arcmin)
        self.sigma_beam = self.beam_size_fwhm / np.sqrt(8.*np.log(2.)) # fwhm to sigma  

    def transfer_function(self, L):
        if self.det=='SPT3G' or self.det=='SPT3G_main' or self.det=='SPT3G_winter':
            return np.exp(-((6000-L)/6600)**6)*np.exp(-(L/8000)**6)
        elif self.det=='Planck':
            return np.ones(len(L))

    def get_tfalm(self, ell, lmin, lmax, mmin, bl=False):
        """
        Returns the transfer function alm. If bl is True, 
        the object returned is the TF already convolved with
        the beam.
        """
        tf = self.transfer_function(ell)
        beam = self.beam_function(ell) if bl else [None]
        tfalm = np.array([grid2alm(almtf2d(tf, lmax, lmin=lmin, mmin=mmin, bl=b)) for b in beam])
        return tfalm

    def beam_function(self, L):
        # return np.exp(-0.5 * np.outer(L**2, self.sigma_beam**2)).T
        return np.exp(-0.5 * np.outer(L*(L+1), self.sigma_beam**2)).T

    def noise_eff(self, L):
        if not isinstance(L, np.ndarray):
            L = np.array(L)

        noise_eff = np.zeros([self.nb, len(L)])
        beam = self.beam_function(L)
        for b in np.arange(self.nb):
            noise_eff[b] = (np.square(self.white_noise[b])*(1+(L/self.l_knee[b])**self.alpha[b]))/np.square(beam[b]) # in muK^2-radians^2  
        noise_eff = np.where(np.isinf(noise_eff), 0, noise_eff)

        return noise_eff
    
    def load_bandpass(self, freq):

        # Frequency in GHz
        nu = (np.arange(40000)+1)
    
        if self.det=='SPT3G' or self.det == 'SPT3G_main' or self.det == 'SPT3G_winter':

            if int(freq) in [95, 150, 220]:
                # f     = f'../input/spt3g_bandpass_{freq}GHz_2019.txt' # up-to-date SPT3G bandpasses
                # ffile = os.path.abspath(os.path.join(os.getcwd(), f))
                # with open(ffile, 'r') as file:
                #     lines = [line.split() for line in file.readlines()[5:]]
                # ff = np.array([float(line[0]) for line in lines])
                # tt = np.array([float(line[1]) for line in lines])

                # f     = f'../input/2019_fts_science_bands.pkl' # up-to-date SPT3G bandpasses
                # ffile = os.path.abspath(os.path.join(os.getcwd(), f))
                # with open(ffile, 'rb') as file:
                #     data = pickle.load(file)
                # if freq == 95:
                #     freq = 90
                # ff = data['bandpasses']['focal_plane_average'][str(freq)]['freq']
                # tt = data['bandpasses']['focal_plane_average'][str(freq)]['spectrum']

                f     = f'../input/focal_plane_averaged_spectra.pkl' # bandpasses in Agora sims, DO NOT use for SPT3G analysis
                base_dir = os.path.dirname(os.path.abspath(__file__))
                ffile = os.path.join(base_dir, f)
                tmp   = pickle.load( open(ffile, "rb" ) )
                if freq == 95:
                    freq = 90
                ff    = tmp[str(freq)]['real_sky_correct']['freq']
                tt    = tmp[str(freq)]['real_sky_correct']['spec']
                tt[tt < 0] = 0
                bpass = np.interp(nu,ff,tt,left=0,right=0)
            else:
                sys.exit('frequency must be 95/150/220 for spt3g')
        
        if self.det == 'Planck':
            if freq in [100, 143, 217, 353, 545, 857]:
                f     = '../input/HFI_RIMO_R3.00.fits'
                base_dir = os.path.dirname(os.path.abspath(__file__))                
                ffile = os.path.join(base_dir, f)
                d     = fits.open(ffile)
                ff = d[f'BANDPASS_F{freq}'].data['WAVENUMBER'] * 3e8 * 1e-7  # Wavenumber in units of cm^-1
                tt = d[f'BANDPASS_F{freq}'].data['TRANSMISSION']
                bpass = np.interp(nu,ff,tt,left=0,right=0)
            else:
                sys.exit('frequency must be 100/143/217/353/545/857 for planck')
        
        return nu,bpass
