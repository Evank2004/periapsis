from .orbit import Orbit
from orbit_package.utils.solvers import solve_kepler
import numpy as np

class CampbellOrbit(Orbit):
    def __init__(self, P, a, e, cosi, omega, Omega, t0, velocity_ratio=None,ref_epoch=None, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0):
        super().__init__(
            P=P,
            a=a, e=e, cosi=cosi, 
            omega=omega, Omega=Omega, 
            t0=t0, velocity_ratio=velocity_ratio,
            ref_epoch=ref_epoch,
            dx=dx, dy=dy, dpmra=dpmra, dpmdec=dpmdec,)
        
        cO = np.cos(Omega)
        sO = np.sin(Omega)
        cw = np.cos(omega)
        sw = np.sin(omega)

        self.A = a * (cO * cw - sO * sw * cosi)
        self.B = a * (sO * cw + cO * sw * cosi)
        self.F = a * (-cO * sw - sO * cw * cosi)
        self.G = a * (-sO * sw + cO * cw * cosi)

    def astrometry(self, t):
        ref_epoch = self.params.get('ref_epoch', None)
        if ref_epoch is None:
            ref_epoch = 0.0
        dt = np.asarray(t) - ref_epoch
        ti = dt - self.params['t0'] * self.params['P']

        M = 2 * np.pi / self.params['P'] * ti
        E = solve_kepler(M, self.params['e'])
        X = (np.cos(E) - self.params['e'])
        Y = (np.sqrt(1 - self.params['e']**2) * np.sin(E))
        alpha = self.A * X + self.F * Y 
        delta = self.B * X + self.G * Y
        
        
        

        alpha = alpha + self.params.get('dx', 0.0) + self.params.get('dpmra', 0.0) * dt
        delta = delta + self.params.get('dy', 0.0) + self.params.get('dpmdec', 0.0) * dt
        return alpha, delta
        
    
    def rv(self, t):
        raise NotImplementedError("The rv method is not implemented yet.")
    
    def xyz(self, t):
        raise NotImplementedError("The xyz method is not implemented yet.")
    
    def vxyz(self, t):
        raise NotImplementedError("The vxyz method is not implemented yet.")

