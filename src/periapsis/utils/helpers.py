from periapsis.model import thieleinnes
from periapsis.model import campbell
import numpy as np

def _build_model(results, params):
    params = _match_param_keys(params)
    fit_method = getattr(results, 'fit_method', None)
    if fit_method is None:
        fit_method = results.samples.get('fit_method', None)

    ref_epoch = getattr(results, 'ref_epoch', None)
    if ref_epoch is None:
        ref_epoch = results.samples.get('ref_epoch', None)

    thiele_innes_keys = {'P', 'e', 't0', 'A', 'B', 'F', 'G'}
    campbell_keys = {'P', 'e', 't0', 'a', 'cosi', 'omega', 'Omega'}

    if fit_method in {'ThieleInnes', 'linear'}:
        if thiele_innes_keys.issubset(params):
            return thieleinnes.ThieleInnesOrbit(ref_epoch=ref_epoch, **params)
        if campbell_keys.issubset(params):
            return campbell.CampbellOrbit(ref_epoch=ref_epoch, **params)

    if fit_method in {'Campbell'}:
        if campbell_keys.issubset(params):
            return campbell.CampbellOrbit(ref_epoch=ref_epoch, **params)
        if thiele_innes_keys.issubset(params):
            return thieleinnes.ThieleInnesOrbit(ref_epoch=ref_epoch, **params)

    if thiele_innes_keys.issubset(params):
        return thieleinnes.ThieleInnesOrbit(ref_epoch=ref_epoch, **params)
    if campbell_keys.issubset(params):
        return campbell.CampbellOrbit(ref_epoch=ref_epoch, **params)

    raise ValueError(f"Unsupported fit method for plotting: {fit_method}")



def _match_param_keys(prior_kwargs):
    '''
    Maps user provided parameter keys to the model constructor names.
    '''


    param_map ={
        'p': 'P','P':'P','period':'P',
        'a':'a','semimajoraxis':'a','semi_major_axis':'a','a1':'a','avis':'a',
        'e':'e','eccentricity':'e',
        'cosi':'cosi',
        'omega':'omega','argperi':'omega','w':'omega',
        'Omega':'Omega','longnode':'Omega','bigomega':'Omega','long':'Omega',
        'T0':'t0','tperi':'t0','timeperi':'t0','tp':'t0','t0':'t0',
        'A':'A','B':'B','F':'F','G':'G',
        'dx':'dx','dy':'dy',
        'dpmra':'dpmra','dmux':'dpmra',
        'dpmdec':'dpmdec','dmuy':'dpmdec'
    }

    normalized = {}

    for user_key, value in prior_kwargs.items():
        if user_key not in param_map:
            raise ValueError(f"Unrecognized parameter key: {user_key}")
        normalized[param_map[user_key]] = value

    return normalized


def _helper_for_periodogram(A,x,err):
    w = 1.0 / err
    x_w = x * w
    A_w = A * w[:, None]

    ATA = A_w.T @ A_w
    ATx = A_w.T @ x_w

    try:
        mu = np.linalg.solve(ATA, ATx)
    except np.linalg.LinAlgError:
        mu, _, _, _ = np.linalg.lstsq(A_w, x_w, rcond=None) # [delta alpha,delta delta, parallax,mu_alpha,mu_delta]


    model_werr = A_w @ mu

    residuals = x_w - model_werr
    chi2 = np.sum(residuals**2)

   

    return mu, chi2
    
