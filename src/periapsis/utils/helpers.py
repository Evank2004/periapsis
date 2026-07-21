from periapsis.model import thieleinnes
from periapsis.model import campbell

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
        'a':'a1','semimajoraxis':'a1','semi_major_axis':'a1','a1':'a1','avis':'a1',
        'e':'e','eccentricity':'e',
        'cosi':'cosi',
        'omega':'omega1','argperi':'omega1','w':'omega1','omega1':'omega1',
        'Omega':'Omega','longnode':'Omega','bigomega':'Omega','long':'Omega',
        't0':'t0','tperi':'t0','timeperi':'t0','tp':'t0','T0':'t0',
        'A':'A1','B':'B1','F':'F1','G':'G1',
        'A1': 'A1','B1':'B1','F1':'F1','G1':'G1',
        'dx':'dx','dy':'dy',
        'dpmra':'dpmra','dmux':'dpmra',
        'dpmdec':'dpmdec','dmuy':'dpmdec',
        'm2':'M2','M2':'M2',
        'm1':'M1','M1':'M1',
    }

    normalized = {}

    for user_key, value in prior_kwargs.items():
        if user_key not in param_map:
            raise ValueError(f"Unrecognized parameter key: {user_key}")
        normalized[param_map[user_key]] = value

    return normalized
    
