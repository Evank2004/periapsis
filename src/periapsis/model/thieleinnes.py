from .orbit import OldOrbit
from periapsis.utils.solvers import solve_kepler
import numpy as np

class ThieleInnesOrbit(OldOrbit):
    def __init__(self, P, e, t0, A1, B1, F1, G1, velocity_ratio=None, ref_epoch=None, dx=0.0, dy=0.0, dpmra=0.0, dpmdec=0.0):
        super().__init__(
            P=P,
            e=e,
            t0=t0,
            A=A1,
            B=B1,
            F=F1,
            G=G1,
            velocity_ratio=velocity_ratio,
            ref_epoch=ref_epoch,
            dx=dx,
            dy=dy,
            dpmra=dpmra,
            dpmdec=dpmdec,
        )

    def astrometry(self, t, system=None):
        ref_epoch = self.params.get('ref_epoch', None)
        if ref_epoch is None:
            ref_epoch = 0.0
        dt = np.asarray(t) - ref_epoch

        M = 2 * np.pi / self.params['P'] * (dt - self.params['t0']*self.params['P'])
        E = solve_kepler(M, self.params['e'])
        X = (np.cos(E) - self.params['e'])
        Y = (np.sqrt(1 - self.params['e']**2) * np.sin(E))
        alpha = self.params['A'] * X + self.params['F'] * Y
        delta = self.params['B'] * X + self.params['G'] * Y

        alpha = alpha + self.params.get('dx', 0.0) + self.params.get('dpmra', 0.0) * dt
        delta = delta + self.params.get('dy', 0.0) + self.params.get('dpmdec', 0.0) * dt
        return alpha, delta
    
    def rv(self, t):
        raise NotImplementedError("The rv method is not implemented yet.")
    
    def xyz(self, t):
        raise NotImplementedError("The xyz method is not implemented yet.")
    
    def vxyz(self, t):
        raise NotImplementedError("The vxyz method is not implemented yet.")

