import numpy as np
import healpy as hp
from itertools import combinations_with_replacement

from .utils import masked_smoothing


class NILC(object):
    """
    Perform the Needlet Internal Linear Combination (NILC) analysis.
    """
    def __init__(self, need, freqs, beams, eff_fwhm=20., beam_thresh=0.01, fwhm_cov=None, mask=None):
        """
        Initialize the NILC class.

        Parameters:
        need (Needlet): The needlet object.
        freqs (array-like): The frequencies to use.
        beams (dict): The beam values for each frequency.
        eff_fwhm (float): The effective FWHM. Default is 20.
        beam_thresh (float or dict): The beam threshold. Default is 0.01.
        fwhm_cov (float or dict): The FWHM to smooth the maps for the covariance estimation. Default is None.
        mask (array-like): The mask to use. Default is None.
        """

        self.need = need
        self.nbands = need.nbands
        self.all_freqs = freqs
        assert(isinstance(beams, dict))
        assert(set(beams.keys()) == set(freqs))
        self.beams = beams
        self.eff_fwhm = eff_fwhm

        if not isinstance(beam_thresh, dict):
            beam_thresh = {nu: beam_thresh for nu in self.all_freqs}
        else:
            for nu in self.all_freqs:
                assert(nu in beam_thresh.keys())
        self.beam_thresh = beam_thresh

        self.beam_ratios = {nu: np.nan_to_num(hp.gauss_beam(np.radians(self.eff_fwhm/60), lmax=self.need.l_maxs[-1])/hp.gauss_beam(np.radians(self.beams[nu]/60), lmax=self.need.l_maxs[-1])) for nu in self.all_freqs}
        self.beam_ratios_inv = {nu: np.nan_to_num(hp.gauss_beam(np.radians(self.beams[nu]/60), lmax=self.need.l_maxs[-1])/hp.gauss_beam(np.radians(self.eff_fwhm/60), lmax=self.need.l_maxs[-1])) for nu in self.all_freqs}
        for nu in self.all_freqs:
            self.beam_ratios[nu][self.beam_ratios[nu] > 1/self.beam_thresh[nu]] = 0
        self.freqs = self.check_freqs_per_band()

        if mask is not None:
            self.mask = {j: hp.ud_grade(mask, nside_out=self.need.nside[j]) for j in range(self.nbands)}
            self.mask_binary = {j: np.zeros_like(self.mask[j]) for j in range(self.nbands)}
            for j in range(self.nbands):
                self.mask_binary[j][self.mask[j] > 0] = 1
        else:
            self.mask = None
            self.mask_binary = None

        if fwhm_cov is None:
            self.fwhm_cov = {j: self.find_fwhm_cov(j) for j in range(self.nbands)}
        else:
            if not isinstance(fwhm_cov, dict):
                fwhm_cov = {j: fwhm_cov for j in range(self.nbands)}
            else:
                for j in range(self.nbands):
                    assert(j in fwhm_cov.keys())
            self.fwhm_cov = fwhm_cov


    def check_freqs_per_band(self):
        """
        Check the frequencies that are good for each band.
        (i.e. the ones that satisfy the beam threshold)
        and return a dictionary with the good frequencies for each band.

        Returns:
        dict: The good frequencies for each band.
        """

        good_freqs = {}
        for j in range(self.nbands):
            good_freqs[j] = []
            for nu in self.all_freqs:
                if ((self.beam_ratios_inv[nu])[self.l_mins[j]:self.l_maxs[j]] > self.beam_thresh[nu]).all():
                    good_freqs[j].append(nu)
        return good_freqs
    

    def Nmodes(self, j, fsky=1):
        """
        Calculate the number of modes in a given band.
        See Eq. 42 from https://arxiv.org/pdf/2307.01043

        Parameters:
        j (int): The band index.
        fsky (float): The fraction of the sky. Default is 1.
        
        Returns:
        float: The number of modes in the given band.
        """
        return (self.h_ell[j]**2*(2*self.l_range+1)).sum()*fsky
    

    def find_fwhm_cov(self, j, N_dep=0, b_tol=0.01, fsky=1):
        """
        See Eq. 43 from https://arxiv.org/pdf/2307.01043
        """
        N_nu = len(self.freqs[j])
        sigma2_real = 2 * (np.abs(1+N_dep-N_nu))/(b_tol*self.Nmodes(j,fsky))
        sigma2_fwhm = sigma2_real * 8*np.log(2)
        return np.rad2deg(np.sqrt(sigma2_fwhm))
    

    def rebeam_alms(self, alm, nu):
        """
        Rebeam the alm coefficients for a given frequency.

        Parameters:
        alm (array): The alm coefficients.
        nu (int): The frequency index.

        Returns:
        array: The rebeamed alm coefficients.
        """
        alm = np.atleast_2d(alm)
        for i in range(alm.shape[0]):
            alm[i] = hp.almxfl(alm[i], self.beam_ratios[nu])
        return alm

    def get_betajk(self, maparr, use_pixel_weights=True, iter=3):
        maparr[maparr==hp.UNSEEN] = 0 # set UNSEEN to 0
        # maparr *= mask
        # TQU -> TEB almarr.shape = (nfreqs, 3, len(alm))
        almarr = np.array([hp.map2alm(maparr[inu], 
                                      lmax=self.need.l_maxs[-1], 
                                      iter=iter, use_pixel_weights=use_pixel_weights) 
                                      for inu in range(len(self.all_freqs))])

        for inu, nu in enumerate(self.all_freqs):
            almarr[inu] = self.rebeam_alms(almarr[inu], nu)

        betajk = {j: np.array([self.need.alm2betajk(almarr[self.all_freqs.index(nu),i], j) 
                               for nu in self.freqs[j] 
                               for i in range(3)]) 
                               for j in range(self.nbands)}
        return betajk
    

   
    def get_betajk_cov(self, idx, stype='comb', use_pixel_weights=True, iter=3, only_b=True, betajk=None, fwhm_cov=None):
        if fwhm_cov is None:
            fwhm_cov = self.fwhm_cov
            if not isinstance(fwhm_cov, dict):
                fwhm_cov = {j: fwhm_cov for j in range(self.nbands)}
        if betajk is None:
            betajk = self.get_betajk(idx, stype=stype, use_pixel_weights=use_pixel_weights, iter=iter, only_b=only_b)
        # calculate betajk_mean as the smoothed needlet coefficient maps
        betajk_mean = {j: np.array([masked_smoothing(betajk[j][i]*self.mask_binary[j], fwhm=fwhm_cov[j]) for i in range(len(betajk[j]))]) for j in range(self.nbands)}
        #  the covariance is calculated by subtracting these means from the full needlet coefficient maps and multiplying them together, then smoothing the result.
        # betajk_cov = {j: np.array([masked_smoothing((betajk[j][i]-betajk_mean[j][i])*(betajk[j][i]-betajk_mean[j][i]), fwhm=fwhm_cov[j]) for i in range(len(betajk[j]))]) for j in range(self.nbands)}
        betajk_cov = {}
        for j in range(self.nbands):
            betajk_cov[j] = {}
            for nu1,nu2 in combinations_with_replacement(self.freqs[j], 2):
                m1 = betajk[j][self.freqs[j].index(nu1)] - betajk_mean[j][self.freqs[j].index(nu1)]
                m2 = betajk[j][self.freqs[j].index(nu2)] - betajk_mean[j][self.freqs[j].index(nu2)]
                betajk_cov[j][(nu1,nu2)] = masked_smoothing(m1*m2*self.mask_binary[j], fwhm=fwhm_cov[j])
        return betajk_cov
