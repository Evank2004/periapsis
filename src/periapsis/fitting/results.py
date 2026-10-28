import numpy as np
from periapsis.data.common import SystemData
from periapsis.data import JointData
from periapsis.params.units import UnitSystem, get_unit_type, normalize_unit
from periapsis.prior import Prior, FixedPrior, Bounds
from periapsis.params import build_transform_function, covered_parameters, overconstrained_parameters, display_latex
from periapsis.utils.helpers import error_sigfig_latex_string, float_to_scientific_latex_string
from periapsis.stats.stat_funcs import red_chi2


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
        self.map_index = np.argmax(self.lnprob) if self.lnprob is not None else None
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
        median_param_16th_84th = {name: np.percentile(self[name, self.unit_system.get_parameter_unit(name)], [15.865, 84.135]) for name in self.median_params.keys()}
        map_param_values = {name: self.unit_system.convert_from_canonical(name, value) for name, value in self.MAP_params.items()} if self.MAP_params is not None else dict()

        table_rows = ["| Parameter | Median (1-$\sigma$) | MAP | Unit |", "| :--- | :--- | :--- | :--- |"]
        for name in self.median_params.keys():
            sampled = name in self.sampled_param_names
            median_value = median_param_values[name]
            map_value = map_param_values.get(name, None)
            lower_error = median_value - median_param_16th_84th[name][0]
            upper_error = median_param_16th_84th[name][1] - median_value
            
            if lower_error <= 0 or upper_error <= 0:
                error_string = f"{median_value}"
                map_value_string = f"{map_value}" if map_value is not None else "N/A"
            else:
                error_string = error_sigfig_latex_string(median_value, upper_error, lower_error)
                map_value_string = error_sigfig_latex_string(map_value, upper_error, lower_error, show_errors=False) if map_value is not None else "{\\rm N/A}"
                
            unit_string = self.unit_system.get_parameter_unit(name).to_string("latex_inline")
            
            table_rows.append(f"| ${display_latex(name)}${' (sampled)' if sampled else ''} | ${error_string}$ | ${map_value_string}$ | {unit_string} |")

        median_params_latex = "\n".join(table_rows)
        red_chi2_map, red_chi2_med,uwe_map,uwe_med,orb_dof = red_chi2(self, self.data)

        other_params_list = ", ".join([f"${display_latex(name)}$" for name in sorted(self.covered_params) if not name.startswith("log") and name not in self.median_params])

        return "\n".join((
            f"### Fit Results ({self.fit_method} {self.backend} on {type(self.data).__name__})",
            f"",
            f"$\\chi^2_{{\\rm red., med}} = {red_chi2_med:.6f}$, $\\chi^2_{{\\rm red., MAP}} = {red_chi2_map:.6f}$",
            f"",
            f"({orb_dof} degrees of freedom)",
            f"",
            f"#### Parameters and Uncertainties",
            f"",
            f"{median_params_latex}",
            f"",
            f"#### Other Available Parameters",
            f"Other available parameters (access with `fit_results['param_name']`): {other_params_list}"
        ))


    def __repr__(self):
        return "\n".join((
            f"FitResults(",
            f"  backend={self.backend},",
            f"  fit_method={self.fit_method},",
            f"  data={type(self.data).__name__ if self.data is not None else None},",
            f"  MAP_params={self.MAP_params},",
            f"  median_params={self.median_params},",
            f")"
        ))


    def __getitem__(self, key) -> ParameterSamples:
        if isinstance(key, tuple):
            key, unit = key
            should_transform_prior = True
            # Automatically transform length parameters to the angular version if the unit requested is an angle.
            if self.unit_system is not None and unit is not None:
                if (get_unit_type(unit) == "angle" or get_unit_type(unit) == "angular_velocity" or get_unit_type(unit) == "angular_acceleration") and not key.endswith("_ang"):
                    key = f"{key}_ang"
        elif self.unit_system is None:
            unit = None
            should_transform_prior = False
        else:
            unit = self.unit_system.get_parameter_unit(key)
            should_transform_prior = False
        

        if key in self.samples:
            if unit is None:
                return ParameterSamples(np.atleast_1d(self.samples[key]), key, None, map_index=self.map_index)
            return ParameterSamples(np.atleast_1d(self.unit_system.convert_from_canonical(key, self.samples[key], unit=unit)), key, unit, map_index=self.map_index)
        
        if self.priors is not None and key in self.priors:
            if isinstance(self.priors[key], FixedPrior):
                prior_unit = self.priors[key].unit
                if prior_unit and not should_transform_prior:
                    return ParameterSamples(np.atleast_1d(self.priors[key].value), key, prior_unit)
                if prior_unit and should_transform_prior:
                    return ParameterSamples(np.atleast_1d(
                        self.unit_system.convert_from_canonical(key, self.canonical_priors[key].value, unit=unit)
                    ), key, unit)
                if not prior_unit:
                    if unit is None:
                        return ParameterSamples(np.atleast_1d(self.priors[key].value), key, None, map_index=self.map_index)
                    return ParameterSamples(np.atleast_1d(
                        self.unit_system.convert_from_canonical(key, self.canonical_priors[key].value, unit=unit)
                    ), key, unit)
            
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
                return ParameterSamples(np.atleast_1d(value), key, None, map_index=self.map_index)
            return ParameterSamples(np.atleast_1d(self.unit_system.convert_from_canonical(key, value, unit=unit)), key, unit, map_index=self.map_index)

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


class ParameterSamples:
    """
    Thin wrapper around a NumPy array of parameter samples, providing convenient access to the samples and their unit.
    """
    def __init__(self, samples, name, unit, map_index=None):
        self.samples = samples
        self.name = name
        self.unit = normalize_unit(unit) if unit is not None else None
        self.map_index = map_index

    def __array__(self):
        return self.samples

    def __array_interface__(self):
        return self.samples.__array_interface__

    def _repr_markdown_(self):
        """Return a markdown representation of the parameter samples."""

        if len(self.samples) == 1:
            return f"${display_latex(self.name)} = {float_to_scientific_latex_string(self.samples[0])}$ {self.unit.to_string('latex_inline') if self.unit is not None else ''} (fixed)"

        v1sigma = 0.682689492137086 * 100
        v2sigma = 0.954499736103642 * 100
        v3sigma = 0.997300203936740 * 100
        p3sl, p2sl, p1sl, p50, p1su, p2su, p3su = np.percentile(self.samples, [50 - v3sigma / 2, 50 - v2sigma / 2, 50 - v1sigma / 2, 50, 50 + v1sigma / 2, 50 + v2sigma / 2, 50 + v3sigma / 2])

        return "\n".join((
            f"${display_latex(self.name)}$ Posterior Samples (n={len(self.samples)}):",
            f"`{np.array2string(self.samples, threshold=5)}` {self.unit.to_string('latex_inline')}" if self.unit is not None else "",
            f"",
            f"\nMedian = {p50} {self.unit.to_string('latex_inline') if self.unit is not None else ''}",
            f"\nMAP    = {self.samples[self.map_index]} {self.unit.to_string('latex_inline') if self.unit is not None else ''}" if self.map_index is not None else "",
            f"\n1-sigma confidence interval: ${error_sigfig_latex_string(p50, p1su - p50, p50 - p1sl)}$ {self.unit.to_string('latex_inline') if self.unit is not None else ''}",
            f"\n2-sigma confidence interval: ${error_sigfig_latex_string(p50, p2su - p50, p50 - p2sl)}$ {self.unit.to_string('latex_inline') if self.unit is not None else ''}",
            f"\n3-sigma confidence interval: ${error_sigfig_latex_string(p50, p3su - p50, p50 - p3sl)}$ {self.unit.to_string('latex_inline') if self.unit is not None else ''}",
        ))

    def __repr__(self):
        return f"{self.name}: {self.samples.__repr__()} <{self.unit}>"

    def __gt__(self, other):
        return self.samples > other

    def __lt__(self, other):
        return self.samples < other

    def __eq__(self, other):
        return self.samples == other

    def __hash__(self):
        return hash((self.samples, self.unit))

    def __ne__(self, other):
        return self.samples != other

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, key):
        return self.samples[key]

    def __contains__(self, item):
        return item in self.samples

    def __getattr__(self, name):
        return getattr(self.samples, name)

    def to_quantity(self):
        """Returns the samples as an astropy Quantity object."""
        if self.unit is None:
            return self.samples
        return self.samples * self.unit

    def to(self, unit):
        """Returns the samples converted to the specified unit."""
        return ParameterSamples(self.unit.to(unit, self.samples), self.name, unit, map_index=self.map_index)
  

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