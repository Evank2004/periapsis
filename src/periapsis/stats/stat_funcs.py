import json

import numpy as np
from periapsis.data.data import Data
from periapsis.data.common import AstrometryData, RadialVelocityData
from periapsis.data.gaia import GaiaData
from periapsis.model.orbit import Orbit
from periapsis.prior import FixedPrior
from scipy.stats import chi2


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _credible_interval_summary(samples):
    m2sig, m1sig, p1sig, p2sig = np.percentile(samples, [2.275, 15.865, 84.135, 97.725])
    return {
        '-2sigma': m2sig,
        '-1sigma': m1sig,
        '+1sigma': p1sig,
        '+2sigma': p2sig,
    }

def red_chi2(results,data,savepath=None):
    '''
    Returns reduced Chi2 value for the MAP and median fit
    '''

    map_params = getattr(results, 'MAP_params', None)
    if map_params is None:
        map_params = results.samples.get('MAP_params', None)
    map_params = dict(map_params) if map_params is not None else {}
    
    med_params = getattr(results, 'median_params', None)
    if med_params is None:
        med_params = results.samples.get('median_params', None)
    med_params = dict(med_params) if med_params is not None else {}

    if map_params is None or med_params is None:
        raise ValueError("Both MAP and median parameter sets are required for reduced Chi2 calculation.")

    num_free_params = len(map_params) - len([p for p in map_params if isinstance(results.priors.get(p), FixedPrior)])

    if not isinstance(data,(GaiaData)):
        map_model = Orbit(**map_params)
        med_model = Orbit(**med_params)

        chi2_map = data.chi2(map_model)
        chi2_med = data.chi2(med_model)
        orbit_dof = data.dof - num_free_params
    else:
        if "jitter" not in map_params: 
            jit = getattr(results, 'jitter', None)
            if jit is None:
                jit = results.samples.get('jitter', None)
            if jit is not None:
                map_params['jitter'] = jit
                med_params['jitter'] = jit

        map_model = Orbit(**map_params)
        med_model = Orbit(**med_params)
        chi2_map = GaiaData.chi2(data,map_model)
        chi2_med = GaiaData.chi2(data,med_model)
        orbit_dof = data.dof - num_free_params # for Gaia data, only one dimension is used for chi2 calculation

    
    map_model = Orbit(**map_params)
    med_model = Orbit(**med_params)

    chi2_map = data.chi2(map_model)
    chi2_med = data.chi2(med_model)

    orbit_dof = data.dof - num_free_params



      # degrees of freedom for the fit
    red_chi2_map = chi2_map / orbit_dof
    red_chi2_med = chi2_med / orbit_dof

    uwe_map = np.sqrt(chi2_map /orbit_dof)
    uwe_med = np.sqrt(chi2_med /orbit_dof)

    return red_chi2_map, red_chi2_med,uwe_map,uwe_med,orbit_dof

def delta_chi2(results,data,savepath=None):
    '''
    Returns delta Chi2 value for orbit fit
    - proper motion fit'''

    

    map_params = getattr(results, 'MAP_params', None)
    if map_params is None:
        map_params = results.samples.get('MAP_params', None)
    map_params = dict(map_params) if map_params is not None else {}
    
    med_params = getattr(results, 'median_params', None)
    if med_params is None:
        med_params = results.samples.get('median_params', None)
    med_params = dict(med_params) if med_params is not None else {}

    num_free_params = len(map_params) - len([p for p in map_params if isinstance(results.priors.get(p), FixedPrior)])

    
    map_model = Orbit(**map_params)
    med_model = Orbit(**med_params)

    if not isinstance(data,(GaiaData,RadialVelocityData)):
        

        pm_chi2 = results.PM_fit['chi2']
        pm_dof = results.PM_fit['dof']
        chi2_map = data.chi2(map_model)
        chi2_med = data.chi2(med_model)
        orbit_dof = 2*len(data.t) - num_free_params

        delta_chi2_map = pm_chi2 - chi2_map #if delta_chi2 > 0, orbit fit is better
        delta_chi2_med = pm_chi2 - chi2_med

        delta_dof_map = np.abs(pm_dof - orbit_dof)
        delta_dof_med = np.abs(pm_dof - orbit_dof)
        
        p_value_map = chi2.sf(delta_chi2_map, delta_dof_map)
        p_value_med = chi2.sf(delta_chi2_med, delta_dof_med) #0.0027 is 3 sigma significance

        sig_significance = (pm_chi2 - pm_dof) / np.sqrt(2*pm_dof) #sigma significance of orbit fit over proper motion fit

    elif isinstance(data,(GaiaData)):
        if "jitter" not in map_params: 
            jit = getattr(results, 'jitter', None)
            if jit is None:
                jit = results.samples.get('jitter', None)
            if jit is not None:
                map_params['jitter'] = jit
                med_params['jitter'] = jit    
        chi2_map = GaiaData.chi2(data,map_model)
        chi2_med = GaiaData.chi2(data,med_model)
        orbit_dof = len(data.t) - num_free_params

        single_chi2 = results.Single_motion_params['chi2']
        single_dof = results.Single_motion_params['dof']

        delta_chi2_map = single_chi2 - chi2_map 
        delta_chi2_med = single_chi2 - chi2_med
        delta_dof_map = np.abs(single_dof - orbit_dof)
        delta_dof_med = np.abs(single_dof - orbit_dof)

        p_value_map = chi2.sf(delta_chi2_map, delta_dof_map)
        p_value_med = chi2.sf(delta_chi2_med, delta_dof_med)

        sig_significance = (single_chi2 - single_dof) / np.sqrt(2*single_dof) #sigma significance of orbit fit over single motion fit

    elif isinstance(data,(RadialVelocityData)):
        gamma_chi2 = results.gamma_fit['chi2']
        gamma_dof = results.gamma_fit['dof']
        chi2_map = data.chi2(map_model)
        chi2_med = data.chi2(med_model)
        orbit_dof = len(data.t) - num_free_params

        delta_chi2_map = gamma_chi2 - chi2_map
        delta_chi2_med = gamma_chi2 - chi2_med
        delta_dof_map = np.abs(gamma_dof - orbit_dof)
        delta_dof_med = np.abs(gamma_dof - orbit_dof)

        p_value_map = chi2.sf(delta_chi2_map, delta_dof_map)
        p_value_med = chi2.sf(delta_chi2_med, delta_dof_med)

        sig_significance = (gamma_chi2 - gamma_dof) / np.sqrt(2*gamma_dof) 

    return delta_chi2_map, delta_chi2_med, p_value_map, p_value_med,sig_significance
    
def credible_intervals(results):
    '''
    Computes the +- 1 and 2 sigma credible intervals for each parameter
    '''

    param_names = results.param_names
    credible_intervals={}
    
    for label in param_names:
        samples = getattr(results, label, None)
        if samples is None:
            samples = results.samples.get(label, None)
        if samples is None:
            raise ValueError(f"Samples for parameter '{label}' not found in results.")

        
        

        credible_intervals[label] = _credible_interval_summary(samples)

    if 'M2' in results.samples:
        credible_intervals['M2'] = _credible_interval_summary(results.samples['M2'])

    return credible_intervals

def all_stats(results,data,pretty_print=True,indent=4,savepath=None):
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
        'credible_intervals': intervals
    }
    if getattr(results, 'backend', None) == 'emcee':
        stats.update({
            'Ess': results.Ess,
            'mean_acceptance_fraction': results.mean_acceptance_fraction,
            'tau': results.tau
        })

    fit_results = {
        'fit_params':{
            'MAP_params': getattr(results, 'MAP_params', None),
            'median_params': getattr(results, 'median_params', None)
        }
    }

    if results['M2'] is not None:
        fit_results['derived_fit_params'] = {
            'M2': {
                'median': float(np.median(results['M2'])),
                'credible_intervals': _credible_interval_summary(results['M2']),
            
            }
        }

    if pretty_print:
        print(json.dumps(stats, indent=indent, sort_keys=True, default=_json_default))
        print("\n" + "=" * 80 + "\n")
        print(json.dumps(fit_results, indent=indent, sort_keys=True, default=_json_default))

    if savepath is not None:
        with open(savepath/"stats.json", "w") as f:
            json.dump(stats, f, indent=indent, sort_keys=True, default=_json_default)
        with open(savepath/"fit_results.json", "w") as f:
            json.dump(fit_results, f, indent=indent, sort_keys=True, default=_json_default)
    

    return stats,fit_results
    
