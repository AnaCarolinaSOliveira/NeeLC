import numpy as np
import healpy as hp
from itertools import combinations_with_replacement

from utils import masked_smoothing, f_sz, f_cib


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
        self.npix = hp.nside2npix(need.nside) # does this always work? might not when input betajk calculated by another instance of needlet
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
        self.nfreqs = {j: len(self.freqs[j]) for j in self.freqs}

        if mask is not None:
            self.mask = {j: hp.ud_grade(mask, nside_out=self.need.nside[j]) for j in range(self.nbands)}
            self.mask_binary = {j: np.zeros_like(self.mask[j]) for j in range(self.nbands)}
            for j in range(self.nbands):
                self.mask_binary[j][self.mask[j] > 0] = 1
        else:
            self.mask = None
            self.mask_binary = {j: 1.0 for j in range(self.nbands)}

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
                if ((self.beam_ratios_inv[nu])[self.need.l_mins[j]:self.need.l_maxs[j]] > self.beam_thresh[nu]).all():
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
        return (self.need.h_ell[j]**2*(2*self.need.l_range+1)).sum()*fsky
    

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

   
    def get_betajk_cov(self, betajk=None, fwhm_cov=None, idx=None, stype='comb', use_pixel_weights=True, iter=3, only_b=True):
        if fwhm_cov is None:
            fwhm_cov = self.fwhm_cov
            if not isinstance(fwhm_cov, dict):
                fwhm_cov = {j: fwhm_cov for j in range(self.nbands)}

        if betajk is None:
            betajk = self.get_betajk(idx, stype=stype, use_pixel_weights=use_pixel_weights, iter=iter, only_b=only_b)

        # calculate betajk_mean as the smoothed needlet coefficient maps
        betajk_mean = {j: np.array([masked_smoothing(betajk[j][nu]*self.mask_binary[j], fwhm=fwhm_cov[j]) for nu in range(len(self.freqs[j]))]) for j in range(self.nbands)}
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
    

    def cov2mat(self, cov):
        """
        Recasts get_betajk_cov() covariance matrix 
        from hashtable to array.
        """
        betajk_mat = {}
        for j in range(self.nbands):
            betajk_mat[j] = np.zeros([self.nfreqs[j],self.nfreqs[j],self.npix[j]])
            for nu1,nu2 in combinations_with_replacement(self.freqs[j], 2):
                betajk_mat[j][self.freqs[j].index(nu1)][self.freqs[j].index(nu2)] = cov[j][(nu1, nu2)]
                betajk_mat[j][self.freqs[j].index(nu2)][self.freqs[j].index(nu1)] = cov[j][(nu1, nu2)]

        return betajk_mat
    
    def get_seds(self, cmb, tsz, cib):
        nc = sum([cmb, tsz, cib])
        seds = {j: np.zeros([self.npix[j], self.nfreqs[j], nc]) for j in range(self.nbands)}

        t_cmb_muk = 2.726e6 # muK
        
        def tsz_sed(nus):
            tszfac = t_cmb_muk * f_sz(nus) # tSZ spectral dependence, in compton y units 
            tszfac_norm = tszfac / 1 # tszfac[1] 
            return tszfac_norm
        
        def cib_sed(nus):
            beta_p = 1.48
            beta_cl = 2.23
            cibfac = t_cmb_muk * f_cib(nus, beta_p) # using poisson term only, for now
            cibfac_norm = cibfac / 1 # cibfac[1]
            return cibfac_norm
        
        for j in range(self.nbands):
            freqs = np.array(self.freqs[j], dtype=int)
            idx=0
            if cmb:
                seds[j][:,:,idx] = 1.
                idx+=1

            if tsz:
                tszfac_norm = tsz_sed(freqs)
                for b in range(len(freqs)):
                    seds[j][:,b,idx] = tszfac_norm[b]
                idx+=1

            if cib:
                cibfac_norm = cib_sed(freqs)
                for b in range(len(freqs)):
                    seds[j][:,b,idx] = cibfac_norm[b]
                idx+=1
        
        return seds 

    def get_weights(self, betajk_cov=None, seds=None, cmb=True, tsz=False, cib=False):
        nc = sum([cmb, tsz, cib])

        if betajk_cov is None:
            betajk_cov = self.get_betajk_cov()
            covmat = self.cov2mat(betajk_cov)
        else:
            if isinstance(betajk_cov[0], dict):
                covmat = self.cov2mat(betajk_cov)
            else: 
                covmat = betajk_cov

        if seds is None: 
            seds = self.get_seds(cmb=cmb, tsz=tsz, cib=cib)
 
        weights = {j: np.zeros([self.npix[j],nc,self.nfreqs[j]]) for j in range(self.nbands)}
        noise_pred = {j: np.zeros([self.npix[j],nc,]) for j in range(self.nbands)}

        for j in range(self.nbands):
            c = np.transpose(covmat[j])
            for pix in range(self.npix[j]):
                cinv = np.linalg.inv(c[pix])
                atc = np.dot(np.transpose(seds[j][pix]),cinv) 
                atca = np.dot(atc,seds[j][pix])
                iatca = np.linalg.inv(atca)
                if nc > 1:
                    noise_pred[j][pix] = np.asarray([np.sqrt(iatca[i,i]) for i in np.arange(nc)])
                else:
                    noise_pred[j][pix] = np.sqrt(iatca)
            
                weights[j][pix] = np.dot(iatca,atc)

        return weights, noise_pred
    
    def plot_weights(self, weights):
        pass

    def separate_betajk(self, betajk, weights=None, use_pixel_weights=True):
        # if no weights are provided, the MV weights for CMB are going to be set as default

        if weights is None:
            # for this to work, we need to fix get_betajk()
            weights = self.get_weights(betajk_cov=None, seds=None, cmb=True, tsz=False, cib=False)

        nc = weights[0].shape[1] # find a better solution for the number of components?

        separated_bjk = {}
        for c in range(nc):
            separated_bjk[c] = 0
            for j in range(self.nbands):
                bjk = np.transpose(betajk[j])
                w_j = weights[j][:,c] 
                T_j_ilc = np.array([np.dot(w_j[pix], bjk[pix]) for pix in range(len(w_j))]) 
                alm_j = hp.map2alm(T_j_ilc, lmax=self.need.l_maxs[j], use_pixel_weights=use_pixel_weights)
                hxalm_j = hp.almxfl(alm_j, self.need.h_ell[j,:])
                z_j = hp.alm2map(hxalm_j, nside=max(self.need.nside))
                separated_bjk[c] += z_j

        return separated_bjk
        
