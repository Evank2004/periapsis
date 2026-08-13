import numpy as np
from periapsis.utils.helpers import _lsq_helper, _matrix_builder, _matrix_filler, _null_matrix_builder, _fill_periodogram_periodic
from periapsis.data.gaia import GaiaData
from periapsis.params.transforms import build_transform_functions

from .initial import InitialGuess


class GaiaInitialGuess(InitialGuess):
    """Class for obtaining initial guess for Gaia data"""
    def __init__(self, data, rng: np.random.RandomState, ref_epoch=0.0, **priors):
        super().__init__(data, rng, ref_epoch, **priors)
        self.M_base, self.cols = _matrix_builder(data, ref_epoch)
        

    def Delisle_periodogram(self,num_freq=10000):
        """Compute the Delisle periodogram to obtain an initial guess on Period"""

        prior_p = self.priors.get('P')
        p_min = prior_p.min if prior_p is not None else 0.001
        p_max = prior_p.max if prior_p is not None else 100 

        M_null,_ = _null_matrix_builder(self.data,self.ref_epoch)


        _,chi2H = _lsq_helper(M_null,self.data.x,self.data.err)

        min_freq = 1/p_max
        max_freq = 1/p_min
        frequencies = np.logspace(np.log10(min_freq),np.log10(max_freq),num_freq)
        periods = 1/frequencies
        power = np.zeros(num_freq)
  

        for i, nu in enumerate(frequencies):
            phase = 2 * np.pi * nu 
            M = self.M_base.copy()
            _fill_periodogram_periodic(M, self.cols,self.ref_epoch,self.data,phase)

            _,chi2K = _lsq_helper(M,self.data.x,self.data.err)

            z_GLS = (chi2H - chi2K) / chi2H

            power[i] = z_GLS

        P_guess = periods[np.argmax(power)]
        max_pwr = power[np.argmax(power)]

        return P_guess, max_pwr
    

    def get_initial_guess(self, param_order, nwalkers):
        "Returns Period guess from Delisle periodogram and random samples of other parameters"
        P_guess, _ = self.Delisle_periodogram()
        initial = []
        for i in self.priors:
            if i == 'P':
                initial.append(P_guess)
            else:
                prior = self.priors[i]
                initial.append(self.rng.uniform(prior.min,prior.max))

        initial_fit_priors = dict(zip(self.priors.keys(),initial))
        transform = build_transform_functions(initial_fit_priors.keys(), ('P', 'e', 'Tp',))
        initial_fit = transform(**initial_fit_priors)
    

        P0 = initial_fit['P']
        e0 = initial_fit['e']
        Tp0 = initial_fit['Tp']
        initial_set = []
        for name in param_order:
            if name == "jitter":
                initial_set.append(0.005)
            elif name not in initial_fit:
                raise ValueError(f"Missing initial guess for parameter: {name}")
            else:
                initial_set.append(initial_fit[name])       
        
        # initial_params = np.clip(np.asarray(initial_set, dtype=float), lower, upper)
        initial_params = np.asarray(initial_set, dtype=float)
                
        # initial = np.clip(initial_params + np.random.randn(self.nwalkers,len(param_order)) * 1e-2, lower, upper)
        initial = initial_params + self.rng.normal(size=(nwalkers,len(param_order))) * 1e-2 * initial_params
        return initial
