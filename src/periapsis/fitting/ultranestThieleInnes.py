from .fitter import Fitter
from periapsis.data.data import Data
from periapsis.fitting.results import FitResults
from periapsis.utils.solvers import transform_theile
from periapsis.model import thieleinnes
from periapsis.utils.helpers import _match_param_keys
from periapsis.prior import FixedPrior, Bounds
import numpy as np
import ultranest

class UltraNestThieleInnes(Fitter):
    def __init__(self, nlive, min_ess, m1=None, **priors):
        super().__init__(m1=m1, **priors)
        self.nlive = nlive
        self.min_ess = min_ess

    def fit(self, data: Data) -> FitResults:

        param_order = list(self.prior_kwargs.keys())
        ndim = len(param_order)

        def prior_transform(cube):
            # UltraNest requires tranforming a unit hypercube
            for i, name in enumerate(param_order):
                prior = self.prior_kwargs.get(name)
                if prior is not None:
                    cube[i] = prior.unp(cube[i])
                else:
                    raise ValueError(f"Missing prior transformer for parameter: {name}")
            return cube
        
        def log_likelihood(params):
            params_dict = _match_param_keys(dict(zip(param_order, params)))
            ref_epoch = getattr(data, 'ref_epoch', None)
            model = thieleinnes.ThieleInnesOrbit(ref_epoch=ref_epoch, **params_dict)
            return -0.5 * data.chi2(model)

        pm_fit = self._proper_motion_fit(data)

        sampler = ultranest.ReactiveNestedSampler(param_order, log_likelihood, prior_transform)
        results = sampler.run(min_num_live_points=self.nlive,DKL=np.inf, min_ess=self.min_ess)

        samples = np.array(results['samples'])
        logl = np.array(results['logl'])

        best_i = np.argmax(logl)
        best_params = dict(zip(param_order, samples[best_i]))
        median_params = dict(zip(param_order, np.median(samples, axis=0)))
        

        results_dict = {}
        for i, name in enumerate(param_order):
            results_dict[name] = samples[:, i]
        
        
        results_dict['ESS'] = results['ess']
        results_dict['logZ'] = results['logz']
        results_dict['logZerr'] = results['logzerr']
        results_dict['param_names'] = param_order
        results_dict['raw_sampler'] = sampler
        results_dict['MAP_params'] = best_params
        results_dict['median_params'] = median_params
        results_dict['PM_fit'] = pm_fit
        results_dict['ref_epoch'] = getattr(data, 'ref_epoch', None)
        results_dict['logl'] = logl
        results_dict['samples'] = samples
        results_dict['backend'] = 'ultranest'
        results_dict['fit_method'] = 'ThieleInnes'
        results_dict['priors'] = self.prior_kwargs
        if results_dict['ref_epoch'] is not None:
            results_dict['priors']['Tepoch'] = FixedPrior(results_dict['ref_epoch'])
        if self.m1 is not None and 'M1' not in results_dict['priors']:
            results_dict['priors']['M1'] = FixedPrior(self.m1)

        fit_results = FitResults(**results_dict)
        fit_results.add_mass_samples(m1=self.m1)
        return fit_results