from .fitter import Fitter
from periapsis.data.data import Data
from periapsis.fitting.results import FitResults
from periapsis.model.campbell import CampbellOrbit
from periapsis.initial.initial import InitialFit
from periapsis.utils.helpers import _match_param_keys
from periapsis.utils.solvers import solve_mass
import numpy as np
import ultranest
from typing import Any, cast

class UltraNestCampbell(Fitter):
    def __init__(self, nlive, min_ess, m1=None, m2_max=None, **priors):
        super().__init__(m1=m1, **priors)
        self.nlive = nlive
        self.min_ess = min_ess
        self.m2_max = m2_max

    def fit(self, data: Data) -> FitResults:

        param_order = list(self.prior_kwargs.keys())
        ndim = len(param_order)

        def prior_transform(cube):
            # UltraNest requires tranforming a unit hypercube
            cube = np.array(cube, copy=True)
            for i, name in enumerate(param_order):
                prior = self.prior_kwargs.get(name)
                if prior is not None:
                    cube[i] = prior.unp(cube[i])
                else:
                    raise ValueError(f"Missing prior transformer for parameter: {name}")
            return cube
        
        def log_likelihood(params):
            params_dict = _match_param_keys(dict(zip(param_order, params)))
            model = CampbellOrbit(ref_epoch=getattr(data, 'ref_epoch', None),**params_dict)
            if self.m1 is not None and self.m2_max is not None:
                m2 = solve_mass(params_dict['a'], params_dict['P'], self.m1)
                if not np.isfinite(m2) or m2 > self.m2_max:
                    return -np.inf
            chi2 = data.chi2(model)
            if chi2 is None:
                return -np.inf
            return -0.5 * chi2
        
        pm_fit = self._proper_motion_fit(data)


        sampler = ultranest.ReactiveNestedSampler(param_order, log_likelihood, prior_transform)
        results = cast(dict[str, Any], sampler.run(
            min_num_live_points=self.nlive,
            min_ess=self.min_ess,
            frac_remain=0.5,
            show_status=False,
            viz_callback=cast(Any, False),
        ))

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
        results_dict['logl'] = logl
        results_dict['samples'] = samples
        results_dict['ref_epoch'] = getattr(data, 'ref_epoch', None)
        results_dict['backend'] = 'ultranest'
        results_dict['fit_method'] = 'Campbell'

        fit_results = FitResults(**results_dict)
        fit_results.add_mass_samples(m1=self.m1)
        return fit_results
        
            