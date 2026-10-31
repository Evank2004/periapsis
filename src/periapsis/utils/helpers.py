from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN

import numpy as np

from periapsis.data.common import AstrometryData, PhysicalAstrometryData, RadialVelocityData
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

def _matrix_builder(data,ref_epoch,linear_params=()):
    
    linear_params = set(linear_params)
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
            nrows += 2 * len(d.t_canonical)
        elif isinstance(d, PhysicalAstrometryData):
            nrows += 2 * len(d.t_canonical)
        elif isinstance(d, RadialVelocityData):
            nrows += len(d.t_canonical)
        elif isinstance(d, GaiaData):
            nrows += len(d.t_canonical)
        else:
            raise ValueError(f"Unsupported data type: {type(d)}")

    for d in data_components:
        system = getattr(d, 'system', None)
        if system == 'relative':
            system = ''
        if isinstance(d, PhysicalAstrometryData):
            key_create(f'dalpha')
            key_create(f'mu_alpha')
            if "dmu_alpha" in linear_params:
                key_create(f'dmu_alpha')
            if d.parallax is True:
                key_create(f'parallax')
            key_create(f'ddelta')
            key_create(f'mu_delta')
            if "dmu_delta" in linear_params:
                key_create(f'dmu_delta')
            key_create(f'B{system}')
            key_create(f'G{system}')
            key_create(f'A{system}')
            key_create(f'F{system}')
        elif isinstance(d, AstrometryData):
            key_create(f'dalpha_ang')
            key_create(f'mu_alpha_ang')
            if "dmu_alpha_ang" in linear_params:
                key_create(f'dmu_alpha_ang')
            if d.parallax is True:
                key_create(f'parallax')
            key_create(f'ddelta_ang')
            key_create(f'mu_delta_ang')
            if "dmu_delta_ang" in linear_params:
                key_create(f'dmu_delta_ang')
            key_create(f'B{system}_ang')
            key_create(f'G{system}_ang')
            key_create(f'A{system}_ang')
            key_create(f'F{system}_ang')
        elif isinstance(d, RadialVelocityData):
            key_create(f'gamma')
            key_create(f'h{system}')
            key_create(f'c{system}')
            if "dgamma" in linear_params:
                key_create(f'dgamma')
        elif isinstance(d, GaiaData):
            key_create(f'dalpha_ang')
            key_create(f'mu_alpha_ang')
            if "dmu_alpha_ang" in linear_params:
                key_create(f'dmu_alpha_ang')
            key_create(f'parallax')
            key_create(f'ddelta_ang')
            key_create(f'mu_delta_ang')
            if "dmu_delta_ang" in linear_params:
                key_create(f'dmu_delta_ang')
            key_create(f'B{system}_ang')
            key_create(f'G{system}_ang')
            key_create(f'A{system}_ang')
            key_create(f'F{system}_ang')

    offsets = isinstance(data,JointData) and data.instrument_offsets is True
    if offsets:
        for name in data.instrument_offset_names():
            key_create(name)

    M = np.zeros((nrows, len(cols)))

    row_idx = 0
    for d in data_components:
        system = getattr(d, 'system', None)
        if isinstance(d, PhysicalAstrometryData):
            n_obs = len(d.t_canonical)
            dt = d.t_canonical - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha']] = 1.0
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha']] = dt
            if d.parallax is True:
                M[row_idx:row_idx+n_obs, cols[f'parallax']] = d.plxf_x
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'ddelta']] = 1.0
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'mu_delta']] = dt
            if d.parallax is True:
                M[row_idx+n_obs:row_idx+2*n_obs, cols[f'parallax']] = d.plxf_y
            if "dmu_alpha" in linear_params:
                M[row_idx:row_idx+n_obs, cols[f'dmu_alpha']] = 0.5*dt**2  
            if "dmu_delta" in linear_params:
                M[row_idx+n_obs:row_idx+2*n_obs, cols[f'dmu_delta']] = 0.5*dt**2
            if offsets:
                instrument = str(d.instrument)
                x_name = f"astro_{instrument}_x_offset"
                y_name = f"astro_{instrument}_y_offset"
                if x_name in cols:
                    M[row_idx:row_idx+n_obs, cols[x_name]] = 1.0
                if y_name in cols:
                    M[row_idx+n_obs:row_idx+2*n_obs, cols[y_name]] = 1.0
            row_idx += 2 * n_obs
        elif isinstance(d, AstrometryData):
            n_obs = len(d.t_canonical)
            dt = d.t_canonical - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha_ang']] = 1.0
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha_ang']] = dt
            if d.parallax is True:
                M[row_idx:row_idx+n_obs, cols[f'parallax']] = d.plxf_x
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'ddelta_ang']] = 1.0
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'mu_delta_ang']] = dt
            if d.parallax is True:
                M[row_idx+n_obs:row_idx+2*n_obs, cols[f'parallax']] = d.plxf_y
            if "dmu_alpha_ang" in linear_params:
                M[row_idx:row_idx+n_obs, cols[f'dmu_alpha_ang']] = 0.5*dt**2  
            if "dmu_delta_ang" in linear_params:
                M[row_idx+n_obs:row_idx+2*n_obs, cols[f'dmu_delta_ang']] = 0.5*dt**2
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
            n_obs = len(d.t_canonical)
            M[row_idx:row_idx+n_obs, cols[f'gamma']] = 1.0
            if "dgamma" in linear_params:
                dt = d.t_canonical - ref_epoch
                M[row_idx:row_idx+n_obs, cols[f'dgamma']] = dt
            if offsets:
                instrument = str(d.instrument)
                rv_name = f"rv_{instrument}_offset"
                if rv_name in cols:
                    M[row_idx:row_idx+n_obs, cols[rv_name]] = 1.0
            row_idx += n_obs
        elif isinstance(d, GaiaData):
            n_obs = len(d.t_canonical)
            dt= d.t_canonical - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha_ang']] = d.spsi
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha_ang']] = dt*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'parallax']] = d.plx_fac 
            M[row_idx:row_idx+n_obs, cols[f'ddelta_ang']] = d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'mu_delta_ang']] = dt*d.cpsi
            if "dmu_alpha_ang" in linear_params:
                M[row_idx:row_idx+n_obs, cols[f'dmu_alpha_ang']] = 0.5*dt**2*d.spsi
            if "dmu_delta_ang" in linear_params:
                M[row_idx:row_idx+n_obs, cols[f'dmu_delta_ang']] = 0.5*dt**2*d.cpsi
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
        if system == 'relative':
            system = ''
        if isinstance(d, PhysicalAstrometryData):
            n_obs = len(d.t_canonical)
            P = params['P']
            e = params['e']
            Tp = params['Tp']
            X,Y,nu = _orbit_coords_nu(P,e,Tp,d.t_canonical)
            f_factor = _photocenter_factor(d, params)
            M[row_idx:row_idx+n_obs, cols[f'B{system}']] = X * f_factor
            M[row_idx:row_idx+n_obs, cols[f'G{system}']] = Y * f_factor
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'A{system}']] = X * f_factor
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'F{system}']] = Y * f_factor
            row_idx += 2 * n_obs
        elif isinstance(d, AstrometryData):
            n_obs = len(d.t_canonical)
            P = params['P']
            e = params['e']
            Tp = params['Tp']
            X,Y,nu = _orbit_coords_nu(P,e,Tp,d.t_canonical)
            f_factor = _photocenter_factor(d, params)
            M[row_idx:row_idx+n_obs, cols[f'B{system}_ang']] = X * f_factor
            M[row_idx:row_idx+n_obs, cols[f'G{system}_ang']] = Y * f_factor
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'A{system}_ang']] = X * f_factor
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'F{system}_ang']] = Y * f_factor
            row_idx += 2 * n_obs
        elif isinstance(d, RadialVelocityData):
            n_obs = len(d.t_canonical)
            P = params['P']
            e = params['e']
            Tp = params['Tp']
            X,Y,nu = _orbit_coords_nu(P,e,Tp,d.t_canonical)
            M[row_idx:row_idx+n_obs, cols[f'h{system}']] = np.cos(nu) + e
            M[row_idx:row_idx+n_obs, cols[f'c{system}']] = np.sin(nu)
            row_idx += n_obs
        elif isinstance(d, GaiaData):
            n_obs = len(d.t_canonical)
            P = params['P']
            e = params['e']
            Tp = params['Tp']
            X,Y,nu = _orbit_coords_nu(P,e,Tp,d.t_canonical)
            f_factor = _photocenter_factor(d, params)
            M[row_idx:row_idx+n_obs, cols[f'B{system}_ang']] = f_factor*X*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'G{system}_ang']] = f_factor*Y*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'A{system}_ang']] = f_factor*X*d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'F{system}_ang']] = f_factor*Y*d.cpsi
            row_idx += n_obs

    

                
def null_matrix_builder(data,ref_epoch):
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
        if isinstance(d, AstrometryData) or isinstance(d, PhysicalAstrometryData):
            nrows += 2 * len(d.t_canonical)
        elif isinstance(d, RadialVelocityData):
            nrows += len(d.t_canonical)
        elif isinstance(d, GaiaData):
            nrows += len(d.t_canonical)
        else:
            raise ValueError(f"Unsupported data type: {type(d)}")

    for d in data_components:
        system = getattr(d, 'system', None)
        if isinstance(d, PhysicalAstrometryData):
            key_create(f'dalpha')
            key_create(f'mu_alpha')
            if d.parallax is True:
                key_create(f'parallax')
            key_create(f'ddelta')
            key_create(f'mu_delta')
        elif isinstance(d, AstrometryData):
            key_create(f'dalpha_ang')
            key_create(f'mu_alpha_ang')
            if d.parallax is True:
                key_create(f'parallax')
            key_create(f'ddelta_ang')
            key_create(f'mu_delta_ang')
        elif isinstance(d, RadialVelocityData):
            key_create(f'gamma')
        elif isinstance(d, GaiaData):
            key_create(f'dalpha_ang')
            key_create(f'mu_alpha_ang')
            key_create(f'parallax')
            key_create(f'ddelta_ang')
            key_create(f'mu_delta_ang')

    offsets = isinstance(data,JointData) and data.instrument_offsets is True
    if offsets:
        for name in data.instrument_offset_names():
            key_create(name)

    M = np.zeros((nrows, len(cols)))

    row_idx = 0
    for d in data_components:
        if isinstance(d, PhysicalAstrometryData):
            n_obs = len(d.t_canonical)
            dt = d.t_canonical - ref_epoch
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
        elif isinstance(d, AstrometryData):
            n_obs = len(d.t_canonical)
            dt = d.t_canonical - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha_ang']] = 1.0
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha_ang']] = dt
            if d.parallax is True:
                M[row_idx:row_idx+n_obs, cols[f'parallax']] = d.plxf_x
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'ddelta_ang']] = 1.0
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'mu_delta_ang']] = dt
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
            n_obs = len(d.t_canonical)
            M[row_idx:row_idx+n_obs, cols[f'gamma']] = 1.0
            if offsets:
                instrument = str(d.instrument)
                rv_name = f"rv_{instrument}_offset"
                if rv_name in cols:
                    M[row_idx:row_idx+n_obs, cols[rv_name]] = 1.0
            row_idx += n_obs
        elif isinstance(d, GaiaData):
            n_obs = len(d.t_canonical)
            dt= d.t_canonical - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha_ang']] = d.spsi
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha_ang']] = dt*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'ddelta_ang']] = d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'mu_delta_ang']] = dt*d.cpsi
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
        if system == 'relative':
            system = ''
        if isinstance(d, AstrometryData):
            n_obs = len(d.t_canonical)
            dt = d.t_canonical - ref_epoch
            cos_phase = np.cos(phase*dt)
            sin_phase = np.sin(phase*dt)
            M[row_idx:row_idx+n_obs, cols[f'B{system}_ang']] = cos_phase
            M[row_idx:row_idx+n_obs, cols[f'G{system}_ang']] = sin_phase
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'A{system}_ang']] = cos_phase
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'F{system}_ang']] = sin_phase
            row_idx += 2 * n_obs
        elif isinstance(d, PhysicalAstrometryData):
            n_obs = len(d.t_canonical)
            dt = d.t_canonical - ref_epoch
            cos_phase = np.cos(phase*dt)
            sin_phase = np.sin(phase*dt)
            M[row_idx:row_idx+n_obs, cols[f'B{system}']] = cos_phase
            M[row_idx:row_idx+n_obs, cols[f'G{system}']] = sin_phase
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'A{system}']] = cos_phase
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'F{system}']] = sin_phase
            row_idx += 2 * n_obs
        elif isinstance(d, RadialVelocityData):
            n_obs = len(d.t_canonical)
            dt = d.t_canonical - ref_epoch
            cos_phase = np.cos(phase*dt)
            sin_phase = np.sin(phase*dt)
            M[row_idx:row_idx+n_obs, cols[f'h{system}']] = cos_phase
            M[row_idx:row_idx+n_obs, cols[f'c{system}']] = sin_phase
            row_idx += n_obs
        elif isinstance(d, GaiaData):
            n_obs = len(d.t_canonical)
            dt= d.t_canonical - ref_epoch
            cos_phase = np.cos(phase * dt)
            sin_phase = np.sin(phase * dt)
            M[row_idx:row_idx+n_obs, cols[f'B{system}_ang']] = cos_phase*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'G{system}_ang']] = sin_phase*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'A{system}_ang']] = cos_phase*d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'F{system}_ang']] = sin_phase*d.cpsi
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
                    t=np.concatenate([existing_data.t_canonical, data.t_canonical]),
                    x=np.concatenate([existing_data.x_canonical, data.x_canonical]),
                    y=np.concatenate([existing_data.y_canonical, data.y_canonical]),
                    x_err=np.concatenate([existing_data.x_err_canonical, data.x_err_canonical]),
                    y_err=np.concatenate([existing_data.y_err_canonical, data.y_err_canonical]),
                    plxf_x=np.concatenate([existing_data.plxf_x, data.plxf_x]),
                    plxf_y=np.concatenate([existing_data.plxf_y, data.plxf_y]),
                    system=system
                )
            elif isinstance(data, RadialVelocityData):
                combined_data[group_key] = RadialVelocityData(
                    t=np.concatenate([existing_data.t_canonical, data.t_canonical]),
                    rv=np.concatenate([existing_data.rv_canonical, data.rv_canonical]),
                    rv_err=np.concatenate([existing_data.rv_err_canonical, data.rv_err_canonical]),
                    system=system
                )
            elif isinstance(data, GaiaData):
                combined_data[group_key] = GaiaData(
                    spsi=np.concatenate([existing_data.spsi, data.spsi]),
                    cpsi=np.concatenate([existing_data.cpsi, data.cpsi]),
                    t=np.concatenate([existing_data.t_canonical, data.t_canonical]),
                    plx_fac = np.concatenate([existing_data.plx_fac, data.plx_fac]),
                    x=np.concatenate([existing_data.x_canonical, data.x_canonical]),
                    err=np.concatenate([existing_data.err_canonical, data.err_canonical]),
                    system=system
                )

    return list(combined_data.values())

def _jitter_name(data):
    if isinstance(data, GaiaData):
        return "astro_jitter"
    if isinstance(data, PhysicalAstrometryData):
        prefix = "astro_phys"
    elif isinstance(data, AstrometryData):
        prefix = "astro"
    elif isinstance(data,RadialVelocityData):
        prefix = "rv"
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    if data.instrument is None:
        return f"{prefix}_jitter"

    return f"{prefix}_{data.instrument}_jitter"

def sigma(data, params_dict, base_sigma):

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
            row_count = 2 * len(component.t_canonical)
        elif isinstance(component, (RadialVelocityData, GaiaData)):
            row_count = len(component.t_canonical)
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
            n_rows = 2 * len(datas.t_canonical)
        elif isinstance(datas, (RadialVelocityData, GaiaData)):
            n_rows = len(datas.t_canonical)
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


def float_to_scientific_latex_string(value, precision=None):
    """
    Format a float as a LaTeX string in scientific notation.
    If precision is specified, it will round the value to that many decimal places.
    """
    if precision is not None:
        value = f"{value:.{precision}e}"
    else:
        value = str(value)

    if 'e' in value:
        base, exponent = value.split('e')
        exponent = int(exponent)
        return f"{base} \\times 10^{{{exponent}}}"
    else:
        return value



def error_sigfig_latex_string(value, err_plus, err_minus, show_errors=True):
    """
    Format a value and asymmetric uncertainties using the PDG significant-figure convention.
    """

    try:
        value, err_plus, err_minus = (Decimal(str(x)) for x in (value, err_plus, err_minus))
    except Exception as exc:
        raise ValueError("value and uncertainties must be finite real numbers") from exc

    if not all(x.is_finite() for x in (value, err_plus, err_minus)):
        raise ValueError("value and uncertainties must be finite real numbers")
    if err_plus <= 0 or err_minus <= 0:
        raise ValueError("uncertainties must be positive")

    def order_of_magnitude(number):
        return number.copy_abs().adjusted()

    def rounding_place_for_error(error):
        """Return the decimal exponent at which an uncertainty is rounded."""
        order = order_of_magnitude(error)
        leading_three = int(error.copy_abs().scaleb(2 - order).to_integral_value(rounding=ROUND_FLOOR))
        # PDG retains two figures for 100-354
        if 100 <= leading_three <= 354:
            return order - 1
        return order

    error_orders = [order_of_magnitude(err_plus), order_of_magnitude(err_minus)]
    rounding_place = min(rounding_place_for_error(error) for error in (err_plus, err_minus))

    value_order = order_of_magnitude(value) if value else rounding_place
    largest_error_order = max(error_orders)
    exponent = value_order
    use_exponent = abs(value_order) >= 4
    if largest_error_order > value_order and (largest_error_order >= 2 or value_order <= -4):
        exponent = largest_error_order + (1 if value_order <= -4 else 0)
        use_exponent = True

    def rounded(number, place, rounding=ROUND_HALF_EVEN):
        result = number.quantize(Decimal(1).scaleb(place), rounding=rounding)
        return result.copy_abs() if result == 0 else result

    if use_exponent:
        scale = Decimal(1).scaleb(exponent)
        decimals = max(0, exponent - rounding_place)
        value_rounded = rounded(value / scale, -decimals)
        plus_rounded = rounded(err_plus / scale, -decimals)
        minus_rounded = rounded(err_minus / scale, -decimals)

        # Keep things consistent when rounding turns 9.9... into 10
        if value_rounded.copy_abs() >= 10:
            value_rounded /= Decimal(10)
            exponent += 1

        if not show_errors:
            return f"{value_rounded:.{decimals}f} \\times 10^{{{exponent}}}"
        return (
            f"{value_rounded:.{decimals}f}"
            f"^{{+{plus_rounded:.{decimals}f}}}"
            f"_{{-{minus_rounded:.{decimals}f}}} \\times 10^{{{exponent}}}"
        )

    decimals = max(0, -rounding_place)
    if not show_errors:
        return f"{rounded(value, rounding_place):.{decimals}f}"
    return (
        f"{rounded(value, rounding_place):.{decimals}f}"
        f"^{{+{rounded(err_plus, rounding_place):.{decimals}f}}}"
        f"_{{-{rounded(err_minus, rounding_place):.{decimals}f}}}"
    )
