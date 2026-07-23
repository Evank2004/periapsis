from .common import SystemData
from periapsis.model.orbit import Orbit
from periapsis.utils.solvers import solve_kepler
from .data import Data
import numpy as np


class GaiaData(Data):
    def __init__(self, spsi,cpsi,t,plx_fac,x,err):
        self.spsi = spsi
        self.cpsi = cpsi
        self.t = t
        self.plx_fac = plx_fac
        self.x = x
        self.err = err


    def chi2(self, orbit: Orbit):
        model_x = orbit.gaia_astrometry(self.t,self.spsi,self.cpsi,self.plx_fac)
        chi2 = np.sum(((self.x - model_x) / self.err) ** 2)
        return chi2

    def _astrometry(self, orbit: Orbit):

        params = orbit.derived_params
        model_x = orbit.gaia_astrometry(self.t,self.spsi,self.cpsi,self.plx_fac)

        #smooth orbit for plotting
        t_smooth = np.linspace(np.min(self.t),np.max(self.t),1000)

        Msmooth = 2 * np.pi / params['P'] * (t_smooth - (params['Tp']*params['P']))
        Esmooth = solve_kepler(Msmooth, params['e'])
        Xsmooth = np.cos(Esmooth) - params['e']
        Ysmooth = np.sqrt(1-params['e']**2)*np.sin(Esmooth)

        ra_orb = params['B']*Xsmooth + params['G']*Ysmooth
        dec_orb = params['A']*Xsmooth + params['F']*Ysmooth

        #--------position decomposition ---------
        res_1d = self.x - model_x

        ra_obs = res_1d * self.spsi
        dec_obs = res_1d * self.cpsi

        #pos for Tp
        ra_peri = params['B']*(1-params['e'])
        dec_peri = params['G']*(1-params['e'])


        #--------linear motion ---------

        plx_ra_smooth = np.interp(t_smooth,self.t,self.plx_fac*self.spsi)
        plx_dec_smooth = np.interp(t_smooth,self.t,self.plx_fac*self.cpsi)

        ra_lin = params['dalpha'] + params['mu_alpha']*t_smooth + plx_ra_smooth*params['parallax']
        dec_lin = params['ddelta'] + params['mu_delta']*t_smooth + plx_dec_smooth*params['parallax']

        ra_full = ra_lin + ra_orb
        dec_full = dec_lin + dec_orb

        ra_sky = params['dalpha'] + params['mu_alpha']*t_smooth + params['parallax']*self.plx_fac*self.spsi + ra_orb
        dec_sky = params['ddelta'] + params['mu_delta']*t_smooth + params['parallax']*self.plx_fac*self.cpsi + dec_orb    


        #--------orbital data points for plotting ---------
        
        M_data = 2 * np.pi / params['P'] * (self.t - (params['Tp']*params['P']))
        E_data = solve_kepler(M_data, params['e'])
        X_data = np.cos(E_data) - params['e']
        Y_data = np.sqrt(1-params['e']**2)*np.sin(E_data)
        
        ra_data_orb = params['B']*X_data + params['G']*Y_data
        dec_data_orb = params['A']*X_data + params['F']*Y_data
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
        