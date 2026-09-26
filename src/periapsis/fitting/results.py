import numpy as np
from astropy import units as u
from periapsis.utils.solvers import transform_theile
from periapsis.utils.solvers import solve_mass
from periapsis.model.orbit import Orbit
from periapsis.prior import Prior, FixedPrior, Bounds
from periapsis.params import build_transform_function, covered_parameters, overconstrained_parameters


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
        self.m1 = samples.pop('m1', None)
        self.mass_function = samples.pop('mass_function', None)
        self.priors = samples.pop('priors', dict())
        self.canonical_priors = samples.pop('canonical_priors', None)
        self.parameter_factors = samples.pop('parameter_factors', {})
        self.Ess = samples.pop('Ess', None)
        self.tau = samples.pop('tau', None)
        self.mean_acceptance_fraction = samples.pop('mean_acceptance_fraction', None)
        self.lnprob = samples.pop('lnprob', None)

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


    def __getitem__(self, key):
        if key in self.samples:
            return self.samples[key]
        if self.param_names is not None and key in self.param_names:
            return self.samples[key]
        
        if self.priors is not None and key in self.priors:
            if isinstance(self.priors[key], FixedPrior):
                return self.priors[key].value
            
        known_params = []
        known_param_values = {}
        if self.param_names is not None:
            known_params.extend(self.param_names)
            known_param_values.update({name: self.samples[name] for name in self.param_names if name in self.samples})
        if self.priors is not None:
            known_params.extend([name for name in self.priors.keys() if isinstance(self.priors[name], FixedPrior)])
            known_param_values.update({name: self.priors[name].value for name in self.priors.keys() if isinstance(self.priors[name], FixedPrior)})

        if known_params:
            canonical_values = {
                name: value * self.parameter_factors.get(name, 1.0)
                for name, value in known_param_values.items()
            }
            transform = build_transform_function(known_params, key)
            value = transform(**canonical_values)
            return value / self.parameter_factors.get(key, 1.0)

    def canonical_MAP_params(self):
        """Return the MAP parameter dictionary in canonical units."""
        if self.MAP_params is None:
            return None
        return {
            name: value * self.parameter_factors.get(name, 1.0)
            for name, value in self.MAP_params.items()
        }

    def canonical_median_params(self):
        """Return the median parameter dictionary in canonical units."""
        if self.median_params is None:
            return None
        return {
            name: value * self.parameter_factors.get(name, 1.0)
            for name, value in self.median_params.items()
        }

    def canonical_sample_array(self):
        """Return the stored sampled parameters in canonical units."""
        if self.param_names is None:
            raise ValueError("Parameter names are required for canonical samples.")
        return np.column_stack([
            np.asarray(self.samples[name]) * self.parameter_factors.get(name, 1.0)
            for name in self.param_names
        ])


    def __contains__(self, key):
        if self.known_params is not None and key in self.known_params:
            return True

        if self.covered_params is not None and key in self.covered_params:
            return True

        return False
        
    def sample_priors(self, random_state, size=1) -> 'SampledPriors':
        if len(self.priors) == 0:
            raise ValueError("No priors are available to sample from.")
        return SampledPriors(self.priors, self.param_names, size, random_state)    

    def add_mass_samples(self, data, m1=None):
        """Derive M2 from canonical posterior orbital samples."""
        if not data.has_astrometry():
            return
        if m1 is None and self.canonical_priors is not None:
            prior = self.canonical_priors.get("M1")
            if isinstance(prior, FixedPrior):
                m1 = prior.value
        if m1 is None or self.canonical_priors is None:
            return

        fixed = {
            name: prior.value
            for name, prior in self.canonical_priors.items()
            if isinstance(prior, FixedPrior)
        }
        masses = []
        for sample in self.canonical_sample_array():
            params = dict(zip(self.param_names, sample))
            params.update(fixed)
            orbit = Orbit(**params)
            a1 = float(orbit["a1"])
            if data.parameter_dimension("a1") == "angle":
                if "distance" not in orbit:
                    masses.append(np.nan)
                    continue
                a1 = (
                    a1 * float(orbit["distance"]) * u.rad * u.pc
                ).to_value(u.AU, equivalencies=u.dimensionless_angles())
            masses.append(solve_mass(a1, float(orbit["P"]), float(m1)))

        self.samples["M2"] = np.asarray(masses)
        self.M2 = self.samples["M2"]
        return self.M2
        

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