import numpy as np
from periapsis.data.common import SystemData
from periapsis.data import JointData
from periapsis.params.units import UnitSystem
from periapsis.prior import Prior, FixedPrior, Bounds
from periapsis.params import build_transform_function, covered_parameters, overconstrained_parameters, display_latex
from periapsis.utils.helpers import error_sigfig_latex_string


class FitResults:
    def __init__(self, **samples):
        self.raw_samples = samples.pop('raw_sampler', None)
        self.backend = samples.pop('backend', None)
        self.fit_method = samples.pop('fit_method', None)
        self.MAP_params = samples.pop('MAP_params', None)
        self.median_params = samples.pop('median_params', None)
        self.null_hypothesis = samples.pop('null_hypothesis', None)
        self.param_names = samples.pop('param_names', None)
        self.sampled_param_names = samples.pop('sampled_param_names', None)
        self.sampler = self.backend
        self.mass_function = samples.pop('mass_function', None)
        self.priors = samples.pop('priors', dict())
        self.canonical_priors = samples.pop('canonical_priors', None)
        self.Ess = samples.pop('Ess', None)
        self.tau = samples.pop('tau', None)
        self.mean_acceptance_fraction = samples.pop('mean_acceptance_fraction', None)
        self.lnprob = samples.pop('lnprob', None)
        self.data = samples.pop('data', None)
        if isinstance(self.data, JointData):
            self.unit_system: UnitSystem = self.data.merged_unit_system
        elif isinstance(self.data, SystemData):
            self.unit_system: UnitSystem = self.data.unit_system
        else:
            self.unit_system: UnitSystem = samples.pop('unit_system', None)

        self.samples = samples
        if self.param_names is not None:
            self.samples.setdefault('param_names', self.param_names)

        if self.priors is None:
            self.priors = dict()

        if self.param_names is None:
            self.param_names = []

        self.fixed_params = {name for name in self.priors.keys() if isinstance(self.priors[name], FixedPrior)}
        self.known_params = {
            *self.param_names,
            *self.fixed_params
        }
        self.covered_params = covered_parameters(self.known_params)


    def _repr_markdown_(self):
        """Returns a pretty markdown representation of the fit results."""
        median_param_values = {name: self.unit_system.convert_from_canonical(name, value) for name, value in self.median_params.items()}
        median_param_16th_84th = {name: np.percentile(self[name, self.unit_system.get_parameter_unit(name)], [16, 84]) for name in self.median_params.keys()}
        median_params_latexs = []
        for name in self.median_params.keys():
            sampled = name in self.sampled_param_names
            median_value = median_param_values[name]
            lower_error = median_value - median_param_16th_84th[name][0]
            upper_error = median_param_16th_84th[name][1] - median_value
            if lower_error <= 0 or upper_error <= 0:
                error_string = f"{median_value}"
            else:
                error_string = error_sigfig_latex_string(median_value, upper_error, lower_error)
            unit_string = self.unit_system.get_parameter_unit(name).to_string("latex_inline")
            median_params_latexs.append(f"- ${display_latex(name)} = {error_string}$ {unit_string} {'(sampled)' if sampled else ''}")
        median_params_latex = "\n".join(median_params_latexs)

        return fr"""
### Fit Results ({self.backend} - {self.fit_method})

#### Median Parameters (1-sigma uncertainties)
{median_params_latex}

Other available parameters: {', '.join([f'${display_latex(name)}$' for name in self.covered_params if not name.startswith("log") and name not in self.median_params])}
        """


    def __repr__(self):
        return "\n".join((
            f"FitResults(",
            f"  backend={self.backend},",
            f"  fit_method={self.fit_method},",
            f"  MAP_params={self.MAP_params},",
            f"  median_params={self.median_params},",
            f")"
        ))


    def __getitem__(self, key):
        if isinstance(key, tuple):
            key, unit = key
            should_transform_prior = True
        elif self.unit_system is None:
            unit = None
            should_transform_prior = False
        else:
            unit = self.unit_system.get_parameter_unit(key)
            should_transform_prior = False
        

        if key in self.samples:
            if unit is None:
                return self.samples[key]
            return self.unit_system.convert_from_canonical(key, self.samples[key], unit=unit)
        
        if self.priors is not None and key in self.priors:
            if isinstance(self.priors[key], FixedPrior):
                prior_unit = self.priors[key].unit
                if prior_unit and not should_transform_prior:
                    return self.priors[key].value
                if prior_unit and should_transform_prior:
                    return self.unit_system.convert_from_canonical(key, self.canonical_priors[key].value, unit=unit)
                if not prior_unit:
                    if unit is None:
                        return self.priors[key].value
                    return self.unit_system.convert_from_canonical(key, self.canonical_priors[key].value, unit=unit)
            
        known_params = []
        known_param_values = {}
        if self.param_names is not None:
            known_params.extend(self.param_names)
            known_param_values.update({name: self.samples[name] for name in self.param_names if name in self.samples})
        if self.canonical_priors is not None:
            known_params.extend([name for name in self.canonical_priors.keys() if isinstance(self.canonical_priors[name], FixedPrior)])
            known_param_values.update({name: self.canonical_priors[name].value for name in self.canonical_priors.keys() if isinstance(self.canonical_priors[name], FixedPrior)})

        if known_params:
            transform = build_transform_function(known_params, key)
            value = transform(**known_param_values)
            if unit is None:
                return value
            return self.unit_system.convert_from_canonical(key, value, unit=unit)

    def __contains__(self, key):
        if self.known_params is not None and key in self.known_params:
            return True

        if self.covered_params is not None and key in self.covered_params:
            return True

        return False
        
    def sample_priors(self, random_state, size=1) -> SampledPriors:
        """Returns a SampledPriors object that can be used to sample from the canonical prior distributions."""
        if self.canonical_priors is None or len(self.canonical_priors) == 0:
            raise ValueError("No priors are available to sample from.")
        return SampledPriors(self.canonical_priors, self.param_names, size, random_state)    
        

class SampledPriors:
    def __init__(self, priors: dict[str, Prior], param_order, size, rng: np.random.RandomState):
        self.priors = priors
        self.param_order = param_order
        self.size = size
        self.rng = rng
        self.non_bound_prior_params = {p for p in priors.keys() if not isinstance(priors[p], Bounds)}
        self.prior_covered_params = covered_parameters(self.non_bound_prior_params)
        self.overconstrained_priors = overconstrained_parameters(self.non_bound_prior_params)

        if len(self.overconstrained_priors) > 0:
            print(f"Warning: Some priors are contradictory: {self.overconstrained_priors}. Please replace at least one of the contradictory priors with a Bounds. Sampling behavior of contradictory priors is undefined.")

        # TODO check that priors do not conflict with each other.

        self.sampled_priors = self._sample_priors(param_order, size, rng)

    def _sample_priors(self, param_order, size: int, rng: np.random.RandomState) -> dict[str, np.ndarray]:
        """
        Samples the prior distributions, and then uses rejection sampling to ensure that the sampled parameters are consistent with any provided bounds.
        """
        non_bound_prior_params = {p for p in self.priors.keys() if not isinstance(self.priors[p], Bounds)}
        prior_covered_params = covered_parameters(self.non_bound_prior_params)

        bad = np.ones(size, dtype=bool)
        pos = np.empty((size, len(param_order)))
        warned = False
        while np.any(bad):
            # Sample initial prior distributions
            values = {}
            for name, prior in self.priors.items():
                if not isinstance(prior, Bounds):
                    values[name] = prior.sample(rng, size=np.sum(bad))
            
            # Transform prior distributions to sampled parameters
            poss = []
            for name in param_order:
                try:
                    transform = build_transform_function(values.keys(), name)
                    poss.append(transform(**values))
                except KeyError:
                    poss.append(np.full(np.sum(bad), np.nan))
            pos[bad] = np.array(poss).T

            # Check bounds
            known_params = []
            known_param_values = {}
            known_params.extend(param_order)
            for i, p in enumerate(param_order):
                known_param_values[p] = pos[bad, i]
            known_params.extend(non_bound_prior_params)
            known_param_values.update({p: values[p] for p in non_bound_prior_params})
            
            new_bad = np.zeros(size, dtype=bool)
            for name, prior in self.priors.items():
                if isinstance(prior, Bounds):
                    if name in param_order:
                        if prior.lower is not None:
                            new_bad[bad] |= pos[bad, param_order.index(name)] < prior.lower
                        if prior.upper is not None:
                            new_bad[bad] |= pos[bad, param_order.index(name)] > prior.upper
                    elif name in prior_covered_params:
                        transform = build_transform_function(known_params, name)
                        val = transform(**known_param_values)
                        if prior.lower is not None:
                            new_bad[bad] |= val < prior.lower
                        if prior.upper is not None:
                            new_bad[bad] |= val > prior.upper
                    else:
                        if not warned:
                            print(f"Warning: Bounds prior for parameter {name} cannot be applied to any sampled parameter.")
                            warned = True
            bad = new_bad
        return pos
    
    def __getitem__(self, key):
        if key in self.param_order:
            return self.sampled_priors[:, self.param_order.index(key)]
        else:
            known_params = []
            known_param_values = {}
            known_params.extend(self.param_order)
            for i, p in enumerate(self.param_order):
                known_param_values[p] = self.sampled_priors[:, i]
            known_params.extend([p for p in self.priors.keys() if isinstance(self.priors[p], FixedPrior)])
            known_param_values.update({p: self.priors[p].value for p in self.priors.keys() if isinstance(self.priors[p], FixedPrior)})
            transform = build_transform_function(known_params, key)
            return transform(**known_param_values)