import numpy as np
from periapsis.utils.solvers import transform_theile
from periapsis.utils.solvers import solve_mass
from periapsis.prior import Prior, FixedPrior, Bounds
from periapsis.params import build_transform_function, covered_parameters, overconstrained_parameters


class FitResults:
    def __init__(self, **samples):
        self.raw_samples = samples.pop('raw_sampler', None)
        self.backend = samples.pop('backend', None)
        self.fit_method = samples.pop('fit_method', None)
        self.MAP_params = samples.pop('MAP_params', None)
        self.median_params = samples.pop('median_params', None)
        self.PM_fit = samples.pop('PM_fit', None)
        self.param_names = samples.pop('param_names', None)
        self.sampler = self.backend
        self.m1 = samples.pop('m1', None)
        self.priors = samples.pop('priors', None)

        self.samples = samples
        if self.param_names is not None:
            self.samples.setdefault('param_names', self.param_names)


    def __getitem__(self, key):
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
            transform = build_transform_function(known_params, key)
            return transform(**known_param_values)
        
    def sample_priors(self, random_state, size=1) -> SampledPriors:
        if self.priors is None:
            raise ValueError("No priors are available to sample from.")
        return SampledPriors(self.priors, self.param_names, size, random_state)    
        

    def add_mass_samples(self, m1=None):
        """Add secondary-mass samples derived from the orbital samples."""
        if m1 is None:
            m1 = self.m1
        if m1 is None:
            return

        param_names = self.param_names or self.samples.get('param_names', [])
        if not param_names:
            return

        period_name = next((name for name in param_names if name in {'P', 'p', 'period', 'Period'}), None)
        a1_name = next((name for name in param_names if name in {'a1', 'a', 'semimajoraxis', 'semi_major_axis'}), None)

        if period_name is None:
            return

        if a1_name is None:
            # Attempt to compute a1 from Thiele-Innes parameters if available
            A_name = next((name for name in param_names if name == 'A'), None)
            B_name = next((name for name in param_names if name == 'B'), None)
            F_name = next((name for name in param_names if name == 'F'), None)
            G_name = next((name for name in param_names if name == 'G'), None)

            if A_name and B_name and F_name and G_name:
                A_samps = self.samples.get(A_name)
                B_samps = self.samples.get(B_name)
                F_samps = self.samples.get(F_name)
                G_samps = self.samples.get(G_name)

                a1_samps, _, _, _ = transform_theile(A_samps, B_samps, F_samps, G_samps)
                self.samples['a1'] = a1_samps
                a1_name = 'a1'
            else:
                return

        P_samps = self.samples.get(period_name)
        a1_samps = self.samples.get(a1_name)
        

        m2_samps = solve_mass(np.asarray(a1_samps, dtype=float), np.asarray(P_samps, dtype=float), float(m1))
        m2_samps = np.where(np.isfinite(m2_samps) & (m2_samps > 0), m2_samps, np.nan)
        self.samples['M2'] = m2_samps
        setattr(self,'M2',m2_samps)


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
                transform = build_transform_function(values.keys(), name)
                poss.append(transform(**values))
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