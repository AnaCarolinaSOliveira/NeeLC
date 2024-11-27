import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import healpy as hp 

from utils import *

PATH_INPUTS = '/scratch/users/anaoliv/component_inputs'

class LC(object):

    def __init__(self, bands, lMin=0, lMax=5000, nL=200, cmb=True, tsz=True, cib=True, blind=False):
        
        # setting up ell-space
        self.lMin = lMin
        self.lMax = lMax
        self.nL = nL
        self.Le = np.around(np.linspace(lMin, lMax, num=(nL+1))).astype(int)
        self.Lc = (self.Le[:-1] + self.Le[1:]) / 2
        self.Lw = np.diff(self.Le)

        # components in the LC analysis 
        self.cmb = cmb
        self.tsz = tsz
        self.cib = cib
        self.comps = []
        if cmb:
            self.comps.append('CMB')
        if tsz:
            self.comps.append('tSZ')
        if cib:
            self.comps.append('CIB')
        self.nc = sum([cmb, tsz, cib])

        # detector specifics
        self.bands = bands
        self.nb = len(self.bands)

        self.blind = blind

    def getSeds(self, cmb, tsz, cib):
        bands = self.bands
        nc = sum([cmb, tsz, cib])
        seds = np.zeros([self.nL, self.nb, nc])

        t_cmb_muk = 2.7255e6 # muK

        def bandpass_corrected_sed(nus, factor_func, beta=None):
            sed = np.zeros(len(nus))
            
            for idx, nu in enumerate(nus):
                if int(nu) in [95, 150, 220]:
                    spt = DetSpecs(det='SPT3G_main')
                    freqs, b = spt.load_bandpass(int(nu))
                elif int(nu) in [100, 143, 217, 353, 545, 857]:
                    planck = DetSpecs(det='Planck')
                    freqs, b = planck.load_bandpass(int(nu))
                dbdt = dBdT(freqs)
                
                if beta is not None:
                    factor = t_cmb_muk * factor_func(freqs, beta)
                else:
                    factor = t_cmb_muk * factor_func(freqs)
                
                factor[np.isnan(factor) | np.isinf(factor)] = 0
                sed[idx] = np.trapz(b * factor * dbdt, freqs) / np.trapz(b * dbdt, freqs)

            return sed

        idx=0
        if cmb:
            seds[:,:,idx] = 1.
            idx+=1

        if tsz:
            tszfac = bandpass_corrected_sed(bands, f_sz)
            for b in range(len(bands)):
                seds[:,b,idx] = tszfac[b]
            idx+=1

        if cib:
            beta_p = 1.48
            beta_cl = 2.23
            cibfac = bandpass_corrected_sed(bands, f_cib, beta=beta_p)
            for b in range(len(bands)):
                seds[:,b,idx] = cibfac[b]
            idx+=1
        
        return seds 
    
    def getCov(self, alms=None, blind=False, cmb=True, tsz=True, cib=True):
        bands = self.bands

        covmat = np.zeros([self.nL,self.nb,self.nb])

        if alms is None:
            if blind:
                spt_cmb_alms = [hp.read_alm(filename=f'{PATH_INPUTS}/mdpl2_spt3g_{int(bands[b])}ghz_lcmbNG_uk_alm_lmax4096.fits') for b in np.arange(self.nb)]
                spt_cib_alms = [hp.read_alm(filename=f'{PATH_INPUTS}/mdpl2_spt3g_{int(bands[b])}ghz_lcibNG_uk_alm_lmax4096.fits') for b in np.arange(self.nb)]
                spt_tsz_alms = [hp.read_alm(filename=f'{PATH_INPUTS}/mdpl2_spt3g_{int(bands[b])}ghz_ltszNGbahamas80_uk_alm_lmax4096.fits ') for b in np.arange(self.nb)]
                spt_noise_alms = [hp.read_alm(filename=f'{PATH_INPUTS}/spt3g_noise_{int(bands[b])}ghz_alm_lmax4096.fits') for b in np.arange(self.nb)]

                total_alms = np.sum(np.stack([spt_noise_alms, spt_cmb_alms, spt_tsz_alms, spt_cib_alms], axis=0), axis=0)

                if self.lMax != 4096:
                    total_alms = np.array([reduce_lmax(total_alms[b], lmax=self.lMax) for b in np.arange(self.nb)])

                for i in np.arange(self.nb):
                    for j in np.arange(self.nb):
                        if i != j:
                            cl = hp.alm2cl(alms1=total_alms[i], alms2=total_alms[j], lmax_out=self.lMax)
                        else:
                            cl = hp.alm2cl(total_alms[i], lmax_out=self.lMax)
                        cl_binned = bin_spectrum(cl, self.Le) 
                        for ell in np.arange(self.nL):
                            covmat[ell,i,j] = cl_binned[ell]

            else:
                import camb
                pars = camb.set_params(H0=67.36, ombh2=0.022, omch2=0.12, mnu=0.06, omk=0, 
                                        tau=0.0544, As=2.1e-9, ns=0.965, halofit_version='mead', lmax=int(self.lMax))
                results = camb.get_results(pars)
                cl_cmb_th = results.get_cmb_power_spectra(pars, lmax=self.lMax, CMB_unit='muK', raw_cl=True)['total'][:,0] # in muK^2
                c_cmb = bin_spectrum(cl_cmb_th, self.Le) # rebinning theory CMB to bin edges for weight calculation

                import pickle
                # getting tSZ and CIB spectra and rebinning 
                with open('./../input/agora_cib_tsz_autocross_spec.pk', 'rb') as file:
                    cib_tsz_data = pickle.load(file)
                cib_tsz_data = {i: {j: bin_spectrum(value, self.Le) for j, value in sub_dict.items()} for i, sub_dict in cib_tsz_data.items()}

                tsz_pwr = cib_tsz_data['tszxtsz']
                cib_pwr = cib_tsz_data['cibxcib']
                tszxcib_pwr = cib_tsz_data['cibxtsz']
                
                noise_alms = [hp.read_alm(filename=f'{PATH_INPUTS}/spt3g_noise_{int(bands[b])}ghz_alm_lmax4096.fits') for b in np.arange(self.nb)]
                noise_pwr = np.array([bin_spectrum(hp.alm2cl(noise_alms[b], lmax_out=self.lMax), self.Le) for b in np.arange(self.nb)]).T

                for l in np.arange(self.nL):
                    for i in np.arange(self.nb):
                        freq_i = int(bands[i])

                        # noise uncorrelated between bands 
                        covmat[l,i,i] += noise_pwr[l,i]

                        for j in np.arange(self.nb):
                            freq_j = int(bands[j])
                            if cmb:
                                # CMB 
                                covmat[l,i,j] += c_cmb[l] 
                            if cib:
                                # CIB x CIB
                                c_cib = cib_pwr[f'{freq_i}x{freq_j}']
                                covmat[l,i,j] += c_cib[l]
                            if tsz:
                                # tSZ x tSZ
                                c_tsz = tsz_pwr[f'{freq_i}x{freq_j}']
                                covmat[l,i,j] += c_tsz[l]
                            if tsz and cib:
                                # CIB x tSZ
                                c_cibtsz = tszxcib_pwr[f'{freq_i}x{freq_j}']
                                covmat[l,i,j] += c_cibtsz[l]
                                c_cibtsz = tszxcib_pwr[f'{freq_j}x{freq_i}']
                                covmat[l,i,j] += c_cibtsz[l]
        
        else:
            if len(alms) != self.nb:
                print("Warning: alms array provided does not match number of frequency bands being used.")

            for i in np.arange(self.nb):
                for j in np.arange(self.nb):
                    if i != j:
                        cl = hp.alm2cl(alms1=alms[i], alms2=alms[j], lmax_out=self.lMax)
                    else:
                        cl = hp.alm2cl(alms[i], lmax_out=self.lMax)
                    cl_binned = bin_spectrum(cl, self.Le) 
                    for ell in np.arange(self.nL):
                        covmat[ell,i,j] = cl_binned[ell]

        return covmat
    

    def weights(self, cov=None):
        f = self.getSeds(cmb=self.cmb, tsz=self.tsz, cib=self.cib)

        if cov is None:
            covmat = self.getCov(blind=self.blind)
        else:
            covmat = cov
        
        bweights = np.zeros([self.nL,self.nc,self.nb])

        for l in range(self.nL):
            
            if np.all(f[l] == 0):
                continue

            non_zero_rows = ~np.all(covmat[l] == 0, axis=1)
            non_zero_cols = ~np.all(covmat[l] == 0, axis=0)
            non_zero_indices = np.where(non_zero_rows & non_zero_cols)[0]

            # If no zero-only rows/columns are found, use the entire matrix
            if len(non_zero_indices) == covmat[l].shape[0]:
                valid_covmat = covmat[l]
                valid_f = f[l]
            else:
                # Otherwise, "chop" to the non-zero submatrix
                valid_covmat = covmat[l][np.ix_(non_zero_indices, non_zero_indices)]
                valid_f = f[l][non_zero_indices,:]

            wmat = np.linalg.inv(valid_covmat)
            atw = np.dot(np.transpose(valid_f),wmat)
            atwa = np.dot(atw,valid_f)
            iatwa = np.linalg.inv(atwa)
            valid_bweights = np.dot(iatwa,atw)

            bweights[l,:, :valid_bweights.shape[1]] = valid_bweights

        return bweights
    
    def separate(self, alm, weights=None, res=11, return_map=True):
        nside = 2**res

        if weights is None:
            weights = self.weights()

        w_unbin = np.zeros((self.Le[-1], weights.shape[1], weights.shape[2]))
        for b in range(len(self.Le)-1):
            start_index = self.Le[b]
            end_index = self.Le[b+1]
            w_unbin[start_index:end_index] = weights[b]
        
        if return_map:
            rec_maps = np.zeros([self.nc, 12*(nside**2)])
        
        rec_alms = np.zeros([self.nc,(alm.shape[1])], dtype='complex128')
        for c in range(self.nc):
            w_c = np.array([w_unbin[l][c] for l in range(len(w_unbin))])
            for b in range(self.nb):
                fl = np.array([w_c[l][b] for l in range(len(w_unbin))])
                rec_alms[c] += hp.almxfl(alm[b], fl)
            if return_map:
                rec_maps[c] = hp.alm2map(rec_alms[c], 2**res)

        if return_map:
            return rec_alms, rec_maps
        else:
            return rec_alms

