from astropy.time import Time
from periapsis.data import AstrometryData,RadialVelocityData,JointData,GaiaData
from periapsis.utils.solvers import solve_kepler
import numpy as np

def _lsq_helper(A,x,err):
    w = 1.0 / err
    x_w = x * w
    A_w = A * w[:, None]

    ATA = A_w.T @ A_w
    ATx = A_w.T @ x_w

    try:
        mu = np.linalg.solve(ATA, ATx)
    except np.linalg.LinAlgError:
        mu, _, _, _ = np.linalg.lstsq(A_w, x_w, rcond=None) 

    model_werr = A_w @ mu

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

    M = np.zeros((nrows, len(cols)))

    row_idx = 0
    for d in data_components:
        system = getattr(d, 'system', None)
        if isinstance(d, AstrometryData):
            n_obs = len(d.t)
            dt = d.t - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha']] = 1.0
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha']] = dt
            M[row_idx:row_idx+n_obs, cols[f'parallax']] = d.plxf_x
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'ddelta']] = 1.0
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'mu_delta']] = dt
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'parallax']] = d.plxf_y
            row_idx += 2 * n_obs
        elif isinstance(d, RadialVelocityData):
            n_obs = len(d.t)
            M[row_idx:row_idx+n_obs, cols[f'gamma']] = 1.0
            row_idx += n_obs
        elif isinstance(d, GaiaData):
            n_obs = len(d.t)
            dt= d.t - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha']] = d.spsi
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha']] = dt*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'parallax']] = d.plx_fac 
            M[row_idx:row_idx+n_obs, cols[f'ddelta']] = d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'mu_delta']] = dt*d.cpsi
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
            M[row_idx:row_idx+n_obs, cols[f'B{system}']] = X
            M[row_idx:row_idx+n_obs, cols[f'G{system}']] = Y
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'A{system}']] = X
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'F{system}']] = Y
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
            M[row_idx:row_idx+n_obs, cols[f'B{system}']] = X*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'G{system}']] = Y*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'A{system}']] = X*d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'F{system}']] = Y*d.cpsi
            row_idx += n_obs

    

def _orbit_coords_nu(P,e,Tp,t):
    '''
    Returns the true anomaly and spatail coordinates of the orbit at time t given P,e,Tp
    '''
    ti = t - Tp
    M = 2*np.pi * ti/P
    E = solve_kepler(M,e)
    X = np.cos(E) - e
    Y = np.sqrt(1.0 - e**2) * np.sin(E)
    nu = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))

    return X,Y,nu

                
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

    M = np.zeros((nrows, len(cols)))

    row_idx = 0
    for d in data_components:
        if isinstance(d, AstrometryData):
            n_obs = len(d.t)
            dt = d.t - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha']] = 1.0
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha']] = dt
            M[row_idx:row_idx+n_obs, cols[f'parallax']] = d.plxf_x
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'ddelta']] = 1.0
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'mu_delta']] = dt
            M[row_idx+n_obs:row_idx+2*n_obs, cols[f'parallax']] = d.plxf_y
            row_idx += 2 * n_obs
        elif isinstance(d, RadialVelocityData):
            n_obs = len(d.t)
            M[row_idx:row_idx+n_obs, cols[f'gamma']] = 1.0
            row_idx += n_obs
        elif isinstance(d, GaiaData):
            n_obs = len(d.t)
            dt= d.t - ref_epoch
            M[row_idx:row_idx+n_obs, cols[f'dalpha']] = d.spsi
            M[row_idx:row_idx+n_obs, cols[f'mu_alpha']] = dt*d.spsi
            M[row_idx:row_idx+n_obs, cols[f'ddelta']] = d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'mu_delta']] = dt*d.cpsi
            M[row_idx:row_idx+n_obs, cols[f'parallax']] = d.plx_fac
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
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
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