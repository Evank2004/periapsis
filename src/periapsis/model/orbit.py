class Orbit():
    """Class representing an orbit defined by a set of orbital elements."""
    def __init__(self, *, velocity_ratio=None, **kwparams):
        """
        A Keplerian orbit is defined by orbital parameters. 7 independent parameters are required to fully specify a 3D orbit, but less may be needed in some contexts.

        velocity_ratio optionally gives the conversion factor to convert from units of (time/distance) to the preferred units for velocity. If none is specified, a warning will be raised whenever a method that computes velocity is called.

        The parameters supported by this class are:
        - P: Orbital period (time)
        - e: Eccentricity (dimensionless)
        - t0: Time of periapsis passage (time)
        - omega: Argument of periapsis (angle)
        - bigomega: Longitude of ascending node (angle)
        - cosi: Cosine of Inclination (angle)
        - a: Semi-major axis (distance)
        - TODO add more parameterizations
        """
        self.params = kwparams
        self.velocity_ratio = velocity_ratio
        self._regularize_params()

    def _regularize_params(self):
        if not self.params:
            raise ValueError("No orbital parameters provided.")

        campbell_params = {'P', 'e', 't0', 'omega', 'Omega', 'cosi', 'a'}
        thiele_innes_params = {'P', 'e', 't0', 'A', 'B', 'F', 'G'}

        if campbell_params.issubset(self.params):
            return

        if thiele_innes_params.issubset(self.params):
            return

        missing_campbell = sorted(campbell_params.difference(self.params))
        missing_thiele = sorted(thiele_innes_params.difference(self.params))
        raise ValueError(
            "Orbital parameters must define either a Campbell orbit "
            f"(missing: {missing_campbell}) or a Thiele-Innes orbit (missing: {missing_thiele})."
        )
        # TODO ensure parameters are sufficient to define an orbit (raise a warning if some information may be missing, e.g. if only 2D information is available rather than the full 3D orbit)
        # TODO add more validation to check for valid parameter values (e.g. e should be between 0 and 1)

    def astrometry(self, t):
        """
        Computes the astrometric position of the orbit at time(s) t. 
        
        Returns the position as a tuple (x, y) in the plane of the sky where x and y are floats or an array of floats matching the shape of t.
        """
        raise NotImplementedError("The astrometry method is not implemented yet.")
        alpha=float('nan')
        delta=float('nan')
        return alpha, delta
    
    def rv(self, t):
        """
        Computes the radial velocity of the orbit at time(s) t. 
        
        Returns the radial velocity as a float or array of floats matching the shape of t.
        """
        raise NotImplementedError("The rv method is not implemented yet.")
        rv=float('nan')
        return rv
    
    def xyz(self, t):
        """
        Computes the 3D position of the orbit at time(s) t, where the x-y plane is the plane of the sky. 
        
        Returns the position as a tuple (x, y, z) where x, y, and z are floats or arrays of floats matching the shape of t.
        """
        raise NotImplementedError("The xyz method is not implemented yet.")
        x=float('nan')
        y=float('nan')
        z=float('nan')
        return x, y, z
    
    def vxyz(self, t):
        """
        Computes the 3D velocity of the orbit at time(s) t, where the x-y plane is the plane of the sky. 
        
        Returns the velocity as a tuple (vx, vy, vz) where vx, vy, and vz are floats or arrays of floats matching the shape of t.
        """
        raise NotImplementedError("The vxyz method is not implemented yet.")
        vx=float('nan')
        vy=float('nan')
        vz=float('nan')
        return vx, vy, vz
        