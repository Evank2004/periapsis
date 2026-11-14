import json
from collections import defaultdict

import numpy as np
from periapsis.data.common import _AstrometryDataBase, AstrometryData, PhysicalAstrometryData, RadialVelocityData
from periapsis.data.data import Data
from periapsis.data import GaiaData, JointData
from periapsis.utils.helpers import _flatten_and_join
import periapsis.utils.helpers as helpers
from periapsis.model.orbit import Orbit
from periapsis.prior import FixedPrior
from scipy.stats import chi2
from periapsis.utils.helpers import error_sigfig_latex_string
from periapsis.params import display_latex
from periapsis.params.transforms import build_transform_functions
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
        samples = results[label, unit]        

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


def _subdata(data, idx):
    if isinstance(data, _AstrometryDataBase):
        return AstrometryData(
            t=data.t[idx],
            x=data.x[idx],
            y=data.y[idx],
            x_err=data.x_err[idx],
            y_err=data.y_err[idx],
            units=data.units,
            system=data.system,
            plxf_x=data.plxf_x[idx] if data.plxf_x is not None else None,
            plxf_y=data.plxf_y[idx] if data.plxf_y is not None else None,
            ref_epoch=data.ref_epoch,
            instrument=data.instrument,
            band=data.band,
        )
    elif isinstance(data, RadialVelocityData):
        return RadialVelocityData(
            t=data.t[idx],
            rv=data.rv[idx],
            rv_err=data.rv_err[idx],
            units=data.units,
            system=data.system,
            ref_epoch=data.ref_epoch,
            instrument=data.instrument,
        )
    elif isinstance(data, GaiaData):
        return GaiaData(
            spsi=data.spsi[idx],
            cpsi=data.cpsi[idx],
            t=data.t[idx],
            plx_fac=data.plx_fac[idx],
            x=data.x[idx],
            err=data.err[idx],
            units=data.units,
            system=data.system,
            ref_epoch=data.ref_epoch,
            band=data.band
        )
    else:
        raise NotImplementedError(f"Resampling not implemented for data type: {type(data).__name__}")
    

def resample_data(data) -> Data:
    if isinstance(data, JointData):
        sizes = []
        for subdata in data.datas:
            sizes.append(len(subdata.t))
        totsize = sum(sizes)
        idx = np.random.choice(totsize, size=totsize, replace=True)
        newdatas = []
        for i, size in enumerate(sizes):
            sub_idx = idx[idx < size]
            idx = idx[idx >= size] - size
            if len(sub_idx) > 0:
                newdatas.append(_subdata(data.datas[i], sub_idx))
        return JointData(
            newdatas,
            instrument_offsets=data.instrument_offsets,
            ref_astrometry_instrument=data.ref_astrometry_instrument,
            ref_rv_instrument=data.ref_rv_instrument
        )
    else:
        idx = np.random.choice(len(data.t), size=len(data.t), replace=True)
        return _subdata(data, idx)


def linear_error_bootstrap(results, data, num_bootstraps=100, rng=np.random.default_rng()):
    """
    Perform a bootstrap on linear parameters to estimate their uncertainties while sampling from the posterior of non-linear parameters.
    """
    sampled_params = results.sampled_param_names
    params = {k: v.value for k, v in results.canonical_priors.items() if isinstance(v, FixedPrior)}
    transform = build_transform_functions(set({*params.keys(), *sampled_params}), ('P', 'e', 'Tp'))
    linear_params = getattr(results.fitter, "linear_params", results.priors.keys())
    bs_params = defaultdict(lambda: [])
    # params.update({name: np.median(results.samples[name]) for name in sampled_params})
    for _ in range(num_bootstraps):
        # Sample from the posterior of non-linear parameters
        sidx = rng.choice(len(results.samples[sampled_params[0]]), size=1)[0]
        params.update({name: results.samples[name][sidx] for name in sampled_params})
        
        PeTp = transform(**params)

        sampled_data = resample_data(data)

        # Matrix method cached variables
        if isinstance(sampled_data, AstrometryData):
            mm_eta = np.concatenate((sampled_data.x_ang_canonical, sampled_data.y_ang_canonical))
            mm_sigma = np.concatenate((sampled_data.x_err_ang_canonical, sampled_data.y_err_ang_canonical)) 
        elif isinstance(sampled_data, PhysicalAstrometryData):
            mm_eta = np.concatenate((sampled_data.x_canonical, sampled_data.y_canonical))
            mm_sigma = np.concatenate((sampled_data.x_err_canonical, sampled_data.y_err_canonical))       
        elif isinstance(sampled_data, RadialVelocityData):
            mm_eta = sampled_data.rv_canonical
            mm_sigma = sampled_data.rv_err_canonical
        elif isinstance(sampled_data,JointData):
            mm_eta = sampled_data._concat_obs_canonical()
            mm_sigma = sampled_data._err_canonical() 
        elif isinstance(sampled_data,GaiaData):
            mm_eta = sampled_data.x_canonical
            mm_sigma = sampled_data.err_canonical

        M, cols = helpers._matrix_builder(sampled_data, sampled_data.ref_epoch_canonical, linear_params)
        helpers._matrix_filler(M, cols, PeTp, sampled_data)
        sig = helpers.sigma(sampled_data, PeTp, mm_sigma)
        mu, chi2 = helpers._lsq_helper(M, mm_eta, sig)
        bs_sample = dict(zip(cols.keys(), mu))
        bs_sample.update(PeTp)
        for k, v in bs_sample.items():
            bs_params[k].append(v)
        bs_params["chi2"].append(chi2)
    
    return bs_params



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
        reported_medians = fit_results["median_params"] or {}
        reported_maps = fit_results["MAP_params"] or {}

        if results.unit_system is not None:
            reported_medians = {
                name: results.unit_system.convert_from_canonical(name, value)
                for name, value in reported_medians.items()
            }
            reported_maps = {
                name: results.unit_system.convert_from_canonical(name, value)
                for name, value in reported_maps.items()
            }
        for label in sorted(intervals):
            unit = (
                results.unit_system.get_parameter_unit(label) if results.unit_system is not None else None
            )
            unit_latex = unit.to_string("latex_inline") if unit is not None else ""

            
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
    
