from .common import SystemData
from periapsis.model.orbit import Orbit
from periapsis.utils.solvers import solve_kepler
from .data import Data
import numpy as np



class GaiaData(SystemData):
    def __init__(self, spsi,cpsi,t,plx_fac,x,err,system=None):
        super().__init__(system)
        self.spsi = np.atleast_1d(spsi)
        self.cpsi = np.atleast_1d(cpsi)
        self.t = np.atleast_1d(t)
        self.plx_fac = np.atleast_1d(plx_fac)
        self.x = np.atleast_1d(x)
        self.err = np.atleast_1d(err)
        if self.x.shape != self.t.shape:
            raise ValueError("x must have the same shape as t")
        if self.spsi.shape != self.t.shape:
            self.spsi = np.broadcast_to(self.spsi, self.t.shape)
        if self.cpsi.shape != self.t.shape:
            self.cpsi = np.broadcast_to(self.cpsi, self.t.shape)
        if self.plx_fac.shape != self.t.shape:
            self.plx_fac = np.broadcast_to(self.plx_fac, self.t.shape)
        if self.err.shape != self.x.shape:
            self.err = np.broadcast_to(self.err, self.x.shape)

    @property
    def dof(self):
        """
        Returns the degrees of freedom of the data.
        """
        return len(self.t)


    def chi2(self, orbit: Orbit,jitter=None):
        model_x = orbit.gaia_astrometry(self.t,self.spsi,self.cpsi,self.plx_fac,system=self.system)
        if "jitter" in orbit.derived_params:
            model_err = np.sqrt(self.err**2 + orbit.derived_params["jitter"]**2)
        else:
            model_err = np.sqrt(self.err**2 + jitter**2) if jitter is not None else self.err
        chi2 = np.sum(((self.x - model_x) / model_err) ** 2)
        return chi2

    def t_series(self):
        """Return the observation timestamps."""
        return self.t

    def _radial_velocity(self, orbit: Orbit):
        """Gaia 1D astrometry dataset has no radial velocity data."""
        return None

    def _astrometry(self, orbit: Orbit):
        system = getattr(self, 'system', None)
        params = orbit.derived_params
        model_x = orbit.gaia_astrometry(self.t,self.spsi,self.cpsi,self.plx_fac,self.system)

        #smooth orbit for plotting
        t_smooth = np.linspace(np.min(self.t),np.max(self.t),1000)

        Msmooth = 2 * np.pi / params['P'] * (t_smooth - (params['Tp']*params['P']))
        Esmooth = solve_kepler(Msmooth, params['e'])
        Xsmooth = np.cos(Esmooth) - params['e']
        Ysmooth = np.sqrt(1-params['e']**2)*np.sin(Esmooth)

        ra_orb = params[f'B{system}']*Xsmooth + params[f'G{system}']*Ysmooth
        dec_orb = params[f'A{system}']*Xsmooth + params[f'F{system}']*Ysmooth

        #--------position decomposition ---------
        res_1d = self.x - model_x

        ra_obs = res_1d * self.spsi
        dec_obs = res_1d * self.cpsi

        #pos for Tp
        ra_peri = params[f'B{system}']*(1-params['e'])
        dec_peri = params[f'A{system}']*(1-params['e'])


        #--------linear motion ---------

        plx_ra_smooth = np.interp(t_smooth,self.t,self.plx_fac*self.spsi)
        plx_dec_smooth = np.interp(t_smooth,self.t,self.plx_fac*self.cpsi)

        ra_lin = params['dalpha'] + params['mu_alpha']*t_smooth + plx_ra_smooth*params['parallax']
        dec_lin = params['ddelta'] + params['mu_delta']*t_smooth + plx_dec_smooth*params['parallax']

        ra_full = ra_lin + ra_orb
        dec_full = dec_lin + dec_orb

        ra_sky = params['dalpha'] + params['mu_alpha']*t_smooth + params['parallax']*plx_ra_smooth + ra_orb
        dec_sky = params['ddelta'] + params['mu_delta']*t_smooth + params['parallax']*plx_dec_smooth + dec_orb    


        #--------orbital data points for plotting ---------
        
        M_data = 2 * np.pi / params['P'] * (self.t - (params['Tp']*params['P']))
        E_data = solve_kepler(M_data, params['e'])
        X_data = np.cos(E_data) - params['e']
        Y_data = np.sqrt(1-params['e']**2)*np.sin(E_data)
        
        ra_data_orb = params[f'B{system}']*X_data + params[f'G{system}']*Y_data
        dec_data_orb = params[f'A{system}']*X_data + params[f'F{system}']*Y_data
        ra_orb_obs = ra_data_orb + ra_obs
        dec_orb_obs = dec_data_orb + dec_obs

        ra_sky_model_data = params['dalpha'] + params['mu_alpha']*self.t + params['parallax']*self.plx_fac*self.spsi + ra_data_orb
        dec_sky_model_data = params['ddelta'] + params['mu_delta']*self.t + params['parallax']*self.plx_fac*self.cpsi + dec_data_orb
        #---------- projected sky data points for plotting ---------
        ra_sky_data = ra_sky_model_data + ra_obs
        dec_sky_data = dec_sky_model_data + dec_obs
       
        return {
            "ra_obs": ra_obs,
            "dec_obs": dec_obs,
            "ra_orb": ra_orb,
            "dec_orb": dec_orb,
            "ra_orb_obs": ra_orb_obs,
            "dec_orb_obs": dec_orb_obs,
            "ra_lin": ra_lin,
            "dec_lin": dec_lin,
            "ra_full": ra_full,
            "dec_full": dec_full,
            "ra_sky": ra_sky,
            "dec_sky": dec_sky,
            "ra_peri": ra_peri,
            "dec_peri": dec_peri,
            "t_smooth": t_smooth,
            "ra_sky_data": ra_sky_data,
            "dec_sky_data": dec_sky_data
        }
        