import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import healpy as hp 

from utils import *


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
        
        spt = DetSpecs(det='SPT')
        self.noise = spt.noise_eff(L=self.Lc)

        self.blind = blind

    def getSeds(self, cmb, tsz, cib):
        bands = self.bands
        seds = np.zeros([self.nL, self.nb, self.nc])

        t_cmb_muk = 2.726e6 # muK

        idx=0
        if cmb:
            seds[:,:,idx] = 1.
            idx+=1

        if tsz:
            tszfac = t_cmb_muk * f_sz(bands) # tSZ spectral dependence, in compton y units 
            tszfac_norm = tszfac #/ tszfac[1] 
            for b in range(len(bands)):
                seds[:,b,idx] = tszfac_norm[b]
            idx+=1

        if cib:
            beta_p = 1.48
            beta_cl = 2.23
            cibfac = t_cmb_muk * f_cib(bands, beta_p) # using poisson term only, for now
            cibfac_norm = cibfac #/ cibfac[1]
            for b in range(len(bands)):
                seds[:,b,idx] = cibfac_norm[b]
            idx+=1
        
        return seds 
    
    def getCov(self, alms=None, noise=None, blind=False, cmb=True, tsz=True, cib=True):
        bands = self.bands

        if alms is None:
            if noise is None:
                noise = self.noise.T

            covmat = np.zeros([self.nL,self.nb,self.nb])
            if blind:
                spt_files = [f'./../input/spt3g_{int(bands[b])}ghz_cmb_tszmasked_cib_noise_alms.fits' for b in np.arange(self.nb)]
                spt_alms = [hp.read_alm(filename=file) for file in spt_files]
                spt = DetSpecs(det='SPT')
                noise_eff = spt.noise_eff(np.arange(5000))
                noise_alms = [hp.synalm(i, lmax=5000) for i in noise_eff]
                total_alms = [spt_alms[b] + noise_alms[b] for b in np.arange(self.nb)]

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
                
                for l in np.arange(self.nL):
                    for i in np.arange(self.nb):
                        freq_i = int(bands[i])

                        # noise uncorrelated between bands 
                        covmat[l,i,i] += noise[l,i]

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
            covmat = np.zeros([self.nL,self.nb,self.nb])
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
 
    def getResiduals(self, weights, cmb, tsz, cib):
        """
        Outputs the power spectrum of the residuals for each of 
        the separated components of the analysis, in the same order
        they appear in the weight matrix. 

        Units of the residual power spectrum is the same as the 
        output alms for the component.
        """
        cov_res = self.getCov(blind=False, cmb=cmb, tsz=tsz, cib=cib)
        cl_res = np.zeros([self.nc,self.nL]) 
        for c in range(weights.shape[1]):
            for l in range(self.nL):
                cl_res[c][l] = np.dot(np.transpose(weights[l][c]), np.dot(cov_res[l], weights[l][c]))

        return cl_res
    

    def weights(self, cov=None, noise=None):
        seds = self.getSeds(cmb=self.cmb, tsz=self.tsz, cib=self.cib)
        if noise is None:
            noise = self.noise.T
 
        if cov is None:
            covmat = self.getCov(noise=noise, blind=self.blind)
        else:
            covmat = cov
        
        bweights = np.zeros([self.nL,self.nc,self.nb])

        for l in range(self.nL):
            wmat = np.linalg.inv(covmat[l])
            atw = np.dot(np.transpose(seds[l]),wmat)
            atwa = np.dot(atw,seds[l])
            iatwa = np.linalg.inv(atwa)
            # if self.nc > 1:
            #     noise_pred[l] = np.asarray([np.sqrt(iatwa[i,i]) for i in np.arange(self.nc)])
            # else:
            #     noise_pred[l] = np.sqrt(iatwa)
        
            bweights[l] = np.dot(iatwa,atw)

        return bweights#, noise_pred
    
    def separate(self, alm, weights=None, res=11, return_map=True):
        nside = 2**res

        if weights is None:
            weights = self.weights(noise=self.noise)


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
        
    
    ##########################
    ######## Plotting ########
    ##########################

    def plotResiduals(self, residual, comp_power, comp, title='', savename=None):
        import camb
        pars = camb.set_params(H0=67.36, ombh2=0.022, omch2=0.12, mnu=0.06, omk=0, 
                                tau=0.0544, As=2.1e-9, ns=0.965, halofit_version='mead', lmax=int(self.lMax))
        results = camb.get_results(pars)
        Dl_cmb_theory = results.get_cmb_power_spectra(pars, lmax=self.lMax, CMB_unit='muK', raw_cl=False)['total'][:,0]
        Lrange = np.linspace(2, self.lMax, len(Dl_cmb_theory)-2)
        Lrange_binned = np.linspace(0, self.lMax, len(residual))

        Dl_res = cl2dl(residual, Lrange=Lrange_binned)
        Dl_comp = cl2dl(comp_power[2:], Lrange=Lrange) 

        plt.figure(figsize=(10,6))
        if comp == "CMB":
            plt.loglog(Lrange, Dl_cmb_theory[2:] , color='gray', linestyle='--', label="CMB theory")
            plt.loglog(Lrange_binned, Dl_res, color='lightcoral', label="expected residuals")
        if comp == 'tSZ':
            tsz_150_alm = hp.read_alm('./../input/spt3g_150ghz_ltszNGbahamas80_uk_diffusioninp_alm.fits')
            tsz_cl = hp.alm2cl(tsz_150_alm, lmax=self.lMax)
            Dl_tsz = cl2dl(tsz_cl[2:], Lrange=Lrange)
            plt.loglog(Lrange, Dl_tsz / (2.726e6 * f_sz(150))**2., color='gray', linestyle='--', label="Compton-y 150Ghz sim")
            plt.loglog(Lrange_binned, Dl_res, color='lightcoral', label="expected residuals")
        plt.loglog(Lrange, Dl_comp, color='black', label=f"reconstructed {comp}")
        plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi}~C_{\ell}$')
        plt.legend()
        plt.title(title)
        plt.xlabel(r'$\ell$')
        if not savename is None:
            plt.savefig(f'./../output/{savename}.png', dpi=600, transparent=False)
        plt.show()
        plt.close()

