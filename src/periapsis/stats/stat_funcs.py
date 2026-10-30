import json

import numpy as np
from periapsis.data.data import Data
from periapsis.data import AstrometryData, RadialVelocityData,GaiaData,JointData
from periapsis.utils.helpers import _flatten_and_join 
from periapsis.model.orbit import Orbit
from periapsis.prior import FixedPrior
from scipy.stats import chi2
from periapsis.utils.helpers import error_sigfig_latex_string
from periapsis.params import display_latex
import astropy.units as u


def _json_numpy(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, u.UnitBase):
        return value.to_string()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def _latex_credible_interval(label, summary, map_value=None, label_suffix="", unit_latex=""):
    median = float(summary["median"])
    lower_error = median - float(summary["-1sigma"])
    upper_error = float(summary["+1sigma"]) - median
    ul = unit_latex.strip("$") if unit_latex else ""
    return f"${{{display_latex(label)}}}{label_suffix} = {error_sigfig_latex_string(median if map_value is None else map_value, upper_error, lower_error)}{f'~{ul}' if unit_latex else ''}$"

def _credible_interval_summary(samples, unit, map_index):
    m2sig, m1sig, med,p1sig, p2sig = np.percentile(samples, [2.275, 15.865,50, 84.135, 97.725])
    return {
        'unit': unit,
        '-2sigma': m2sig,
        '-1sigma': m1sig,
        'median': med,
        'map': samples[map_index],
        '+1sigma': p1sig,
        '+2sigma': p2sig,
    }


def _get_result_value(results, name, default=None):
    if isinstance(results, dict):
        return results.get(name, default)

    if hasattr(results, '__getitem__'):
        try:
            return results[name]
        except (KeyError, TypeError, IndexError):
            pass

    if hasattr(results, 'samples'):
        samples = getattr(results, 'samples')
        if isinstance(samples, dict) and name in samples:
            return samples[name]

    return getattr(results, name, default)


def _summary_params(results, summary_name):
    canonical_name = f"canonical_{summary_name}_params"
    canonical_getter = getattr(results, canonical_name, None)
    if callable(canonical_getter):
        return canonical_getter()
    params = getattr(results, f"{summary_name}_params", None)
    if params is not None:
        return params

    samples = getattr(results, "samples", None)
    if isinstance(samples, dict):
        return samples.get(f"{summary_name}_params")

    return None


def _fit_jitter(params, data):
    if isinstance(data, GaiaData):
        return params.get("astro_jitter", params.get("astro_Gaia_jitter"))
    prefix = "astro" if isinstance(data, AstrometryData) else "rv"
    name = f"{prefix}_jitter" if data.instrument is None else f"{prefix}_{data.instrument}_jitter"
    return params.get(name)


def _model_chi2(model, params, data):
    jitter = _fit_jitter(params, data)
    if isinstance(data, AstrometryData):
        return data.chi2(model, jitter_x=jitter, jitter_y=jitter)
    if isinstance(data, RadialVelocityData):
        return data.chi2(model, jitter=jitter)
    if isinstance(data, GaiaData):
        return data.chi2(model, jitter=jitter)
    return data.chi2(model)


def red_chi2(results,data,savepath=None):
    '''
    Returns reduced Chi2 value for the MAP and median fit
    '''
    datas = _flatten_and_join(data)

    map_params = _summary_params(results, "MAP")
    med_params = _summary_params(results, "median")

    if map_params is None or med_params is None:
        raise ValueError("Both MAP and median parameter sets are required for reduced Chi2 calculation.")

    map_params = dict(map_params)
    med_params = dict(med_params)

    
    num_free_params = len(map_params) - len([p for p in map_params if isinstance(results.priors.get(p), FixedPrior)])
    med_chi2 = []
    map_chi2 = []
    dof = 0
  
    for d in datas:
        
        map_model = Orbit(**map_params)
        med_model = Orbit(**med_params)
        map_chi2.append(_model_chi2(map_model, map_params, d))
        med_chi2.append(_model_chi2(med_model, med_params, d))
        dof += d.dof
            
    map_chi2 = np.sum(map_chi2)
    med_chi2 = np.sum(med_chi2)
    orb_dof = dof - num_free_params

    red_chi2_map = map_chi2 / orb_dof
    red_chi2_med = med_chi2 / orb_dof
    uwe_map = np.sqrt(map_chi2 /orb_dof)
    uwe_med = np.sqrt(med_chi2 /orb_dof)

    return red_chi2_map, red_chi2_med,uwe_map,uwe_med,orb_dof

def delta_chi2(results,data,savepath=None):
    '''
    Returns delta Chi2 value for orbit fit
    - proper motion fit'''

    map_params = _summary_params(results, "MAP")
    med_params = _summary_params(results, "median")

    if map_params is None or med_params is None:
        raise ValueError("Both MAP and median parameter sets are required for delta Chi2 calculation.")

    map_params = dict(map_params)
    med_params = dict(med_params)

    num_free_params = len(map_params) - len([p for p in map_params if isinstance(results.priors.get(p), FixedPrior)])

    datas = _flatten_and_join(data)

    map_chi2 = []
    med_chi2 = []
    dof = 0
    for d in datas:
        map_model = Orbit(**map_params)
        med_model = Orbit(**med_params)
        map_chi2.append(_model_chi2(map_model, map_params, d))
        med_chi2.append(_model_chi2(med_model, med_params, d))
        dof += d.dof

    map_chi2 = np.sum(map_chi2)
    med_chi2 = np.sum(med_chi2)
    orb_dof = dof - num_free_params
    null_chi2 = results.null_hypothesis['chi2']
    null_dof = results.null_hypothesis['dof']
  
    delta_chi2_map = null_chi2 - map_chi2
    delta_chi2_med = null_chi2 - med_chi2

    p_value_map = chi2.sf(delta_chi2_map, np.abs(null_dof - orb_dof))
    p_value_med = chi2.sf(delta_chi2_med, np.abs(null_dof - orb_dof))

    sig_significance = (null_chi2 - null_dof) / np.sqrt(2*null_dof) #sigma significance of orbit fit over null hypothesis fit


    return delta_chi2_map, delta_chi2_med, p_value_map, p_value_med,sig_significance
    
def credible_intervals(results):
    '''
    Computes the +- 1 and 2 sigma credible intervals for each parameter
    '''

    param_names = results.param_names
    credible_intervals={}
    map_index = results.map_index
    
    for label in param_names:
        unit = results.unit_system.get_parameter_unit(label) if results.unit_system is not None else None
        # samples = getattr(results, label, None)
        samples = results[label, unit]
        # if samples is None:
        #     samples = results.samples.get(label, None)
        # if samples is None:
            # raise ValueError(f"Samples for parameter '{label}' not found in results.")

        
        

        credible_intervals[label] = _credible_interval_summary(samples, unit, map_index)

    m2_samples = _get_result_value(results, 'M2')
    m2_unit = results.unit_system.get_parameter_unit('M2') if results.unit_system is not None else None

    if m2_samples is not None:
        credible_intervals['M2'] = _credible_interval_summary(m2_samples, m2_unit, map_index)
    else:
        min_m2_samples = _get_result_value(results, 'minM2')
        min_m2_unit = results.unit_system.get_parameter_unit('minM2') if results.unit_system is not None else None
        if min_m2_samples is not None:
            credible_intervals['minM2'] = _credible_interval_summary(min_m2_samples, min_m2_unit, map_index)

    return credible_intervals

def all_stats(results,data,pretty_print=True,indent=4,savepath=None):
    if hasattr(results, "add_mass_samples") and "M2" not in results.samples:
        results.add_mass_samples(data)
    red_chi2_map, red_chi2_med,uwe_map,uwe_med,orbit_dof = red_chi2(results,data)
    delta_chi2_map, delta_chi2_med,p_map,p_med,sig_significance = delta_chi2(results,data)
    intervals = credible_intervals(results)
    

    stats = {
        'red_chi2_map': red_chi2_map,
        'red_chi2_med': red_chi2_med,
        'dof': orbit_dof,
        'uwe_map': uwe_map,
        'uwe_med': uwe_med,
        'delta_chi2_map': delta_chi2_map,
        'delta_chi2_med': delta_chi2_med,
        'p_value_map': p_map,
        'p_value_med': p_med,
        'sigma_significance': sig_significance,
    }
    if getattr(results, 'backend', None) == 'emcee':
        stats.update({
            'sampled_params': results.sampled_param_names, 
            'Ess': results.Ess,
            'mean_acceptance_fraction': results.mean_acceptance_fraction,
            'tau': results.tau
        })

    fit_results = {
        'MAP_params': getattr(results, 'MAP_params', None),
        'median_params': getattr(results, 'median_params', None),
        'credible_intervals': intervals,
    }

    if pretty_print:
        print(json.dumps(stats, indent=indent, sort_keys=False, default=_json_numpy))
        print("\n" + "=" * 80 + "\n")
        print(json.dumps(fit_results, indent=indent, sort_keys=False, default=_json_numpy))
        print("\n" + "=" * 80 + "\n")
        print("LaTeX Median values with 1-sigma credible intervals:")
        reported_medians = fit_results['median_params'] or {}
        reported_maps = fit_results['MAP_params'] or {}
        for label in sorted(intervals):
            unit_latex = results.unit_system.get_parameter_unit(label).to_string("latex_inline") if results.unit_system is not None else ""
            print(_latex_credible_interval(
                label,
                intervals[label],
                reported_medians.get(label),
                unit_latex=unit_latex
            ))
        print("\nLaTeX MAP values with 1-sigma credible intervals:")
        for label in sorted(set(intervals) & set(reported_maps)):
            unit_latex = results.unit_system.get_parameter_unit(label).to_string("latex_inline") if results.unit_system is not None else ""
            print(_latex_credible_interval(
                label,
                intervals[label],
                map_value=reported_maps[label],
                label_suffix=r"_{{\rm MAP}}",
                unit_latex=unit_latex
            ))

    if savepath is not None:
        with open(savepath/"stats.json", "w") as f:
            json.dump(stats, f, indent=indent, sort_keys=True, default=_json_numpy)
        with open(savepath/"fit_results.json", "w") as f:
            json.dump(fit_results, f, indent=indent, sort_keys=True, default=_json_numpy)
    

    return stats,fit_results
    
