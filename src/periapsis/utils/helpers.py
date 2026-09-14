from astropy.time import Time
import numpy as np

from periapsis.data.common import AstrometryData, RadialVelocityData
from periapsis.data.gaia import GaiaData
from periapsis.data.joint_data import JointData
from periapsis.utils.solvers import _orbit_coords_nu
from periapsis.params import flux_parameter
from periapsis.params.transforms import build_transform_functions, q_f_to_photocenter_factor

def _lsq_helper(M,x,err):
    w = 1.0 / err
    x_w = x * w
    M_w = M * w[:, None]

    MTM = M_w.T @ M_w
    MTx = M_w.T @ x_w

    try:
        mu = np.linalg.solve(MTM, MTx)
    except np.linalg.LinAlgError:
        mu, _, _, _ = np.linalg.lstsq(M_w, x_w, rcond=None) 

    model_werr = M_w @ mu

    residuals = x_w - model_werr
    chi2 = np.sum(residuals**2)

    return mu, chi2

def _flatten_joint(data):
        """
        Flattens a JointData object into a list of its constituent data objects.
        """
        if isinstance(data, JointData):
            return data.datas
        else:
            return [data]

def _matrix_builder(data,ref_epoch):

    data_components = _flatten_joint(data)

    cols = {}
    col_idx = 0

    def key_create(key):
        nonlocal col_idx
        if key not in cols:
            cols[key] = col_idx
            col_idx += 1
        return cols[key]

    # Check that all components have system attribute
    if not isinstance(data, JointData):
        if getattr(data,'system',None) is None:
            raise ValueError("Data object must have a 'system' attribute")
    else:
        for d in data_components:
            if getattr(d,'system',None) is None:
                raise ValueError("Data components must have a 'system' attribute")

    nrows = 0
    for d in data_components:
        if isinstance(d, AstrometryData):
            nrows += 2 * len(d.t)
        elif isinstance(d, RadialVelocityData):
            nrows += len(d.t)
        elif isinstance(d, GaiaData):
            nrows += len(d.t)
        else:
            raise ValueError(f"Unsupported data type: {type(d)}")

    for d in data_components:
        system = getattr(d, 'system', None)
        if isinstance(d, AstrometryData):
            key_create(f'dalpha')
            key_create(f'mu_alpha')
            if d.parallax is True:
                key_create(f'parallax')
            key_create(f'ddelta')
            key_create(f'mu_delta')
            key_create(f'B{system}')
            key_create(f'G{system}')
            key_create(f'A{system}')
            key_create(f'F{system}')
        elif isinstance(d, RadialVelocityData):
            key_create(f'gamma')
            key_create(f'h{system}')
            key_create(f'c{system}')
            if d.rv_trend is True:
                key_create(f'rv_trend')
        elif isinstance(d, GaiaData):
            key_create(f'dalpha')
            key_create(f'mu_alpha')
            key_create(f'parallax')
            key_create(f'ddelta')
            key_create(f'mu_delta')
            key_create(f'B{system}')
            key_create(f'G{system}')
            key_create(f'A{system}')
            key_create(f'F{system}')

    offsets = isinstance(data,JointData) and data.instrument_offsets is True
    if offsets:
        for name in data.instrument_offset_names():
            key_create(name)

    M = np.zeros((nrows, len(cols)))

    row_idx = 0
    for d in data_components:
        system = getattr(d, 'system', None)
        if isinstance(d, AstrometryData):
            n_obs = len(d.t)
            dt = d.t - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha']] = 1.0
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha']] = dt
            if d.parallax is True:
                M[row_idx:row_idx+n_obs, cols[f'parallax']] = d.plxf_x
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'ddelta']] = 1.0
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'mu_delta']] = dt
            if d.parallax is True:
                M[row_idx+n_obs:row_idx+2*n_obs, cols[f'parallax']] = d.plxf_y
            if offsets:
                instrument = str(d.instrument)
                x_name = f"astro_{instrument}_x_offset"
                y_name = f"astro_{instrument}_y_offset"
                if x_name in cols:
                    M[row_idx:row_idx+n_obs, cols[x_name]] = 1.0
                if y_name in cols:
                    M[row_idx+n_obs:row_idx+2*n_obs, cols[y_name]] = 1.0
            row_idx += 2 * n_obs
        elif isinstance(d, RadialVelocityData):
            n_obs = len(d.t)
            M[row_idx:row_idx+n_obs, cols[f'gamma']] = 1.0
            if d.rv_trend is True:
                dt = d.t - ref_epoch
                M[row_idx:row_idx+n_obs, cols[f'rv_trend']] = dt
            if offsets:
                instrument = str(d.instrument)
                rv_name = f"rv_{instrument}_offset"
                if rv_name in cols:
                    M[row_idx:row_idx+n_obs, cols[rv_name]] = 1.0
            row_idx += n_obs
        elif isinstance(d, GaiaData):
            n_obs = len(d.t)
            dt= d.t - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha']] = d.spsi
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha']] = dt*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'parallax']] = d.plx_fac 
            M[row_idx:row_idx+n_obs, cols[f'ddelta']] = d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'mu_delta']] = dt*d.cpsi
            if offsets:
                instrument = str(d.instrument)
                gaia_name = f"astro_{instrument}_offset"
                if gaia_name in cols:
                    M[row_idx:row_idx+n_obs, cols[gaia_name]] = 1.0
            row_idx += n_obs

    return M, cols



def _matrix_filler(M,cols,params,data):

    data_components = _flatten_joint(data)

    row_idx = 0
    for d in data_components:
        system = getattr(d, 'system', None)
        if isinstance(d, AstrometryData):
            n_obs = len(d.t)
            P = params['P']
            e = params['e']
            Tp = params['Tp']
            X,Y,nu = _orbit_coords_nu(P,e,Tp,d.t)
            f_factor = _photocenter_factor(d, params)
            M[row_idx:row_idx+n_obs, cols[f'B{system}']] = X * f_factor
            M[row_idx:row_idx+n_obs, cols[f'G{system}']] = Y * f_factor
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'A{system}']] = X * f_factor
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'F{system}']] = Y * f_factor
            row_idx += 2 * n_obs
        elif isinstance(d, RadialVelocityData):
            n_obs = len(d.t)
            P = params['P']
            e = params['e']
            Tp = params['Tp']
            X,Y,nu = _orbit_coords_nu(P,e,Tp,d.t)
            M[row_idx:row_idx+n_obs, cols[f'h{system}']] = np.cos(nu) + e
            M[row_idx:row_idx+n_obs, cols[f'c{system}']] = np.sin(nu)
            row_idx += n_obs
        elif isinstance(d, GaiaData):
            n_obs = len(d.t)
            P = params['P']
            e = params['e']
            Tp = params['Tp']
            X,Y,nu = _orbit_coords_nu(P,e,Tp,d.t)
            f_factor = _photocenter_factor(d, params)
            M[row_idx:row_idx+n_obs, cols[f'B{system}']] = f_factor*X*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'G{system}']] = f_factor*Y*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'A{system}']] = f_factor*X*d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'F{system}']] = f_factor*Y*d.cpsi
            row_idx += n_obs

    

                
def _null_matrix_builder(data,ref_epoch):
    """
    Builds a null design matrix for the given data
    Used for computing null hypothesis chi-sqaured
    """       
    data_components = _flatten_joint(data)

    cols = {}
    col_idx = 0

    def key_create(key):
        nonlocal col_idx
        if key not in cols:
            cols[key] = col_idx
            col_idx += 1
        return cols[key]

    # Check that all components have system attribute
    if not isinstance(data, JointData):
        if getattr(data,'system',None) is None:
            raise ValueError("Data object must have a 'system' attribute")
    else:
        for d in data_components:
            if getattr(d,'system',None) is None:
                raise ValueError("Data components must have a 'system' attribute")

    nrows = 0
    for d in data_components:
        if isinstance(d, AstrometryData):
            nrows += 2 * len(d.t)
        elif isinstance(d, RadialVelocityData):
            nrows += len(d.t)
        elif isinstance(d, GaiaData):
            nrows += len(d.t)
        else:
            raise ValueError(f"Unsupported data type: {type(d)}")

    for d in data_components:
        system = getattr(d, 'system', None)
        if isinstance(d, AstrometryData):
            key_create(f'dalpha')
            key_create(f'mu_alpha')
            if d.parallax is True:
                key_create(f'parallax')
            key_create(f'ddelta')
            key_create(f'mu_delta')
        elif isinstance(d, RadialVelocityData):
            key_create(f'gamma')
        elif isinstance(d, GaiaData):
            key_create(f'dalpha')
            key_create(f'mu_alpha')
            key_create(f'parallax')
            key_create(f'ddelta')
            key_create(f'mu_delta')

    offsets = isinstance(data,JointData) and data.instrument_offsets is True
    if offsets:
        for name in data.instrument_offset_names():
            key_create(name)

    M = np.zeros((nrows, len(cols)))

    row_idx = 0
    for d in data_components:
        if isinstance(d, AstrometryData):
            n_obs = len(d.t)
            dt = d.t - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha']] = 1.0
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha']] = dt
            if d.parallax is True:
                M[row_idx:row_idx+n_obs, cols[f'parallax']] = d.plxf_x
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'ddelta']] = 1.0
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'mu_delta']] = dt
            if d.parallax is True:
                M[row_idx+n_obs:row_idx+2*n_obs, cols[f'parallax']] = d.plxf_y
            if offsets:
                instrument = str(d.instrument)
                x_name = f"astro_{instrument}_x_offset"
                y_name = f"astro_{instrument}_y_offset"
                if x_name in cols:
                    M[row_idx:row_idx+n_obs, cols[x_name]] = 1.0
                if y_name in cols:
                    M[row_idx+n_obs:row_idx+2*n_obs, cols[y_name]] = 1.0
            row_idx += 2 * n_obs
        elif isinstance(d, RadialVelocityData):
            n_obs = len(d.t)
            M[row_idx:row_idx+n_obs, cols[f'gamma']] = 1.0
            if offsets:
                instrument = str(d.instrument)
                rv_name = f"rv_{instrument}_offset"
                if rv_name in cols:
                    M[row_idx:row_idx+n_obs, cols[rv_name]] = 1.0
            row_idx += n_obs
        elif isinstance(d, GaiaData):
            n_obs = len(d.t)
            dt= d.t - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha']] = d.spsi
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha']] = dt*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'ddelta']] = d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'mu_delta']] = dt*d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'parallax']] = d.plx_fac
            if offsets:
                instrument = str(d.instrument)
                gaia_name = f"astro_{instrument}_offset"
                if gaia_name in cols:
                    M[row_idx:row_idx+n_obs, cols[gaia_name]] = 1.0
            row_idx += n_obs

    return M, cols


def _fill_periodogram_periodic(M,cols,ref_epoch,data,phase):
    """
    Fills the periodic portion of design matrix for Delisle periodogram 
    """

    data_components = _flatten_joint(data)
    

    row_idx = 0
    for d in data_components:
        system = getattr(d, 'system', None)
        if isinstance(d, AstrometryData):
            n_obs = len(d.t)
            dt = d.t - ref_epoch
            cos_phase = np.cos(phase*dt)
            sin_phase = np.sin(phase*dt)
            M[row_idx:row_idx+n_obs, cols[f'B{system}']] = cos_phase
            M[row_idx:row_idx+n_obs, cols[f'G{system}']] = sin_phase
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'A{system}']] = cos_phase
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'F{system}']] = sin_phase
            row_idx += 2 * n_obs
        elif isinstance(d, RadialVelocityData):
            n_obs = len(d.t)
            dt = d.t - ref_epoch
            cos_phase = np.cos(phase*dt)
            sin_phase = np.sin(phase*dt)
            M[row_idx:row_idx+n_obs, cols[f'h{system}']] = cos_phase
            M[row_idx:row_idx+n_obs, cols[f'c{system}']] = sin_phase
            row_idx += n_obs
        elif isinstance(d, GaiaData):
            n_obs = len(d.t)
            dt= d.t - ref_epoch
            cos_phase = np.cos(phase * dt)
            sin_phase = np.sin(phase * dt)
            M[row_idx:row_idx+n_obs, cols[f'B{system}']] = cos_phase*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'G{system}']] = sin_phase*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'A{system}']] = cos_phase*d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'F{system}']] = sin_phase*d.cpsi
            row_idx += n_obs



def _flatten_and_join(data):
    """Flattens joint data to components and combines components of same system and data type"""
    datas = _flatten_joint(data) if isinstance(data, JointData) else [data]

    grouped_data = []
    for data in datas:
        system = getattr(data, 'system', None)
        group_key = (type(data), system)
        grouped_data.append((group_key, data))

    combined_data = {}
    for (group_key, data) in grouped_data:
        if group_key not in combined_data:
            combined_data[group_key] = data
        else:
            existing_data = combined_data[group_key]
            if isinstance(data, AstrometryData):
                combined_data[group_key] = AstrometryData(
                    t=np.concatenate([existing_data.t, data.t]),
                    x=np.concatenate([existing_data.x, data.x]),
                    y=np.concatenate([existing_data.y, data.y]),
                    x_err=np.concatenate([existing_data.x_err, data.x_err]),
                    y_err=np.concatenate([existing_data.y_err, data.y_err]),
                    plxf_x=np.concatenate([existing_data.plxf_x, data.plxf_x]),
                    plxf_y=np.concatenate([existing_data.plxf_y, data.plxf_y]),
                    system=system
                )
            elif isinstance(data, RadialVelocityData):
                combined_data[group_key] = RadialVelocityData(
                    t=np.concatenate([existing_data.t, data.t]),
                    rv=np.concatenate([existing_data.rv, data.rv]),
                    rv_err=np.concatenate([existing_data.rv_err, data.rv_err]),
                    system=system
                )
            elif isinstance(data, GaiaData):
                combined_data[group_key] = GaiaData(
                    spsi=np.concatenate([existing_data.spsi, data.spsi]),
                    cpsi=np.concatenate([existing_data.cpsi, data.cpsi]),
                    t=np.concatenate([existing_data.t, data.t]),
                    plx_fac = np.concatenate([existing_data.plx_fac, data.plx_fac]),
                    x=np.concatenate([existing_data.x, data.x]),
                    err=np.concatenate([existing_data.err, data.err]),
                    system=system
                )

    return list(combined_data.values())

def _jitter_name(data):
    if isinstance(data,(AstrometryData,GaiaData)):
        prefix = "astro"
    elif isinstance(data,RadialVelocityData):
        prefix = "rv"
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    if data.instrument is None:
        return f"{prefix}_jitter"

    return f"{prefix}_{data.instrument}_jitter"

def _sigma(data, params_dict, base_sigma):

    if not isinstance(data,JointData):
        jitter_name = _jitter_name(data)
        jitter = params_dict.get(jitter_name)

        if jitter is None:
            return base_sigma
        return np.sqrt(base_sigma**2 + jitter**2)

    jitters = [params_dict.get(_jitter_name(component)) for component in data.datas]
    if all(jitter is None for jitter in jitters):
        return base_sigma

    sigmas = []
    row_start = 0
    for component, jitter in zip(data.datas, jitters):
        if isinstance(component, AstrometryData):
            row_count = 2 * len(component.t)
        elif isinstance(component, (RadialVelocityData, GaiaData)):
            row_count = len(component.t)
        else:
            raise ValueError(f"Unsupported data type: {type(component)}")

        component_sigma = base_sigma[row_start:row_start + row_count]
        if jitter is not None:
            component_sigma = np.sqrt(component_sigma**2 + jitter**2)
        sigmas.append(component_sigma)
        row_start += row_count

    return np.concatenate(sigmas)

def _jitter_check(data,params_dict):
    """Returns rows whose likelihood has a jitter paramerer"""
    components = data.datas if isinstance(data,JointData) else [data]
    rows_with_jitter = []

    for datas in components:
        has_jitter = _jitter_name(datas) in params_dict

        if isinstance(datas, AstrometryData):
            n_rows = 2 * len(datas.t)
        elif isinstance(datas, (RadialVelocityData, GaiaData)):
            n_rows = len(datas.t)
        else:
            raise ValueError(f"Unsupported data type: {type(datas)}")

        rows_with_jitter.append(np.full(n_rows, has_jitter, dtype=bool))

    return np.concatenate(rows_with_jitter)

def _photocenter_factor(data,params):

    flux_name = flux_parameter(getattr(data,'band',None))

    if flux_name is None or data.system in ('relative',""):
        return 1.0
    if flux_name not in params:
        return 1.0

    if 'q' not in params:
        transform = build_transform_functions(params.keys(), ('q',))
        params.update(transform(**params))

    return q_f_to_photocenter_factor(params['q'],params[flux_name],data.system)