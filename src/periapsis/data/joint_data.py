from typing import List
import numpy as np
from .data import Data
from periapsis.model.orbit import Orbit
from .gaia import GaiaData
from .common import SystemData, AstrometryData, RadialVelocityData


class JointData(Data):
    def __init__(self, datas: List[SystemData], instrument_offsets=False,
                 ref_astrometry_instrument=None, ref_rv_instrument=None):
        self.datas = datas
        self.instrument_offsets = instrument_offsets
        self.ref_astrometry_instrument = ref_astrometry_instrument
        self.ref_rv_instrument = ref_rv_instrument

        self.check_joint_units()

    def instrument_offset_names(self):
        """Return the linear instrument-offset column names for this joint data."""
        if not self.instrument_offsets:
            return []

        names = []
        for data_type, coordinate_names in (
            ((AstrometryData, GaiaData), None),
            (RadialVelocityData, ("rv",)),
        ):
            components = [
                data for data in self.datas
                if isinstance(data, data_type) and data.instrument is not None
            ]
            if not components:
                continue

            counts = {}
            order = []
            for data in components:
                instrument = str(data.instrument)
                if instrument not in counts:
                    counts[instrument] = 0
                    order.append(instrument)
                counts[instrument] += len(data.t)

            reference_instrument = (
                self.ref_astrometry_instrument
                if data_type != RadialVelocityData
                else self.ref_rv_instrument
            )

            if reference_instrument is None:
                reference = max(
                    order,
                    key=lambda instrument: (counts[instrument], -order.index(instrument)),
                )
            else:
                data_label = (
                    "radial-velocity"
                    if data_type == RadialVelocityData
                    else "astrometry"
                    )
                reference = str(reference_instrument)
                if reference not in counts:
                    raise ValueError(
                        f"Reference instrument '{reference_instrument}' has no "
                        f"{data_label} data."
                    )

           
            for instrument in order:
                if instrument == reference:
                    continue
                if data_type != RadialVelocityData and instrument == "Gaia":
                    names.append(f"astro_{instrument}_offset")
                elif data_type != RadialVelocityData:
                    names.append(f"astro_{instrument}_x_offset")
                    names.append(f"astro_{instrument}_y_offset")
                else:
                    names.append(f"rv_{instrument}_offset")

        return names

    def check_joint_units(self):
        """Check that all corresponding datas in joint
        are provided with same units. Raises ValueError if not."""

        if not self.datas or not all(hasattr(data, "unit_system") for data in self.datas):
            return

        ref_t = self.datas[0].unit_system.units['t']

        for data in self.datas[1:]:
            if data.unit_system.units['t'] != ref_t:
                raise ValueError(f"Joint data must have matching time units. Found {data.unit_system.units['t']} and {ref_t}.")

        astro_datas = [data for data in self.datas if isinstance(data, (AstrometryData,GaiaData))]
        if astro_datas:
            ref_x = astro_datas[0].unit_system.units['x']
            for data in astro_datas[1:]:
                if data.unit_system.units['x'] != ref_x:
                    raise ValueError(f"Joint astrometry data must have matching x units. Found {data.unit_system.units['x']} and {ref_x}.")

        rv_datas = [data for data in self.datas if isinstance(data, RadialVelocityData)]
        if rv_datas:
            ref_rv = rv_datas[0].unit_system.units['rv']
            for data in rv_datas[1:]:
                if data.unit_system.units['rv'] != ref_rv:
                    raise ValueError(f"Joint radial velocity data must have matching rv units. Found {data.unit_system.units['rv']} and {ref_rv}.")

            
    def parameter_unit(self,param_name):

        if param_name.startswith("astro_"):
            for data in self.datas:
                if isinstance(data, (AstrometryData,GaiaData)):
                    return data.parameter_unit(param_name)

        if param_name.startswith("rv_"):
            for data in self.datas:
                if isinstance(data, RadialVelocityData):
                    return data.parameter_unit(param_name)


        for data in self.datas:
            try:
                return data.parameter_unit(param_name)
            except KeyError:
                continue

        raise KeyError(f"No unit found for parameter '{param_name}'.")

    def parameter_dimension(self,param_name):

        if param_name.startswith("astro_"):
            for data in self.datas:
                if isinstance(data, (AstrometryData,GaiaData)):
                    return data.parameter_dimension(param_name)

        if param_name.startswith("rv_"):
            for data in self.datas:
                if isinstance(data, RadialVelocityData):
                    return data.parameter_dimension(param_name)

        for data in self.datas:
            try:
                return data.parameter_dimension(param_name)
            except KeyError:
                continue

        raise KeyError(f"No dimension found for parameter '{param_name}'.")

    @property
    def dof(self):
        """Returns the degrees of freedom of the data."""
        total_dof = 0
        for data in self.datas:
            total_dof += data.dof
        return total_dof

    def chi2(self, orbit: Orbit):
        total_chi2 = 0
        for data in self.datas:
            total_chi2 += data.chi2(orbit)
        return total_chi2

    def log_likelihood(self, orbit: Orbit):
        offset_names = (self.instrument_offset_names())

        return sum(
            data.log_likelihood(orbit, offset_names)
            for data in self.datas
        )

    def as_astrometry_data(self):
        ''' Returns AstrometryData object containing all astrometry data of simialar system in Joint Data. 
        '''
        astrometry_datas = [data for data in self.datas if isinstance(data, AstrometryData)]
        if not astrometry_datas:
            raise ValueError("No astrometry data found in the joint data.")

        has_parallax = any(data.parallax for data in astrometry_datas)
        for system in set(data.system for data in astrometry_datas):
            system_datas = [data for data in astrometry_datas if data.system == system]
            t = np.concatenate([data.t for data in system_datas])
            x = np.concatenate([data.x for data in system_datas])
            y = np.concatenate([data.y for data in system_datas])
            x_err = np.concatenate([data.x_err for data in system_datas])
            y_err = np.concatenate([data.y_err for data in system_datas])
            if has_parallax:
                plxf_x = np.concatenate([data.plxf_x 
                        if data.parallax 
                        else np.zeros_like(data.t,dtype=float)
                        for data in system_datas])
                plxf_y = np.concatenate([data.plxf_y 
                        if data.parallax 
                        else np.zeros_like(data.t,dtype=float)
                        for data in system_datas])
            else:
                plxf_x = None
                plxf_y = None

            source = system_datas[0]
            canonical_units = source.unit_system.canonical_units

            units = {"t": canonical_units["t"],
                    "x": canonical_units["x"],
                    "y": canonical_units["y"],
                    "x_err": canonical_units["x_err"],
                    "y_err": canonical_units["y_err"],}
            return AstrometryData(t, x, y, x_err, y_err,plxf_x,plxf_y, system=system,units =units)
        
    def as_gaia_data(self):
        '''
        Returns GaiaData object containing all Gaia data of simialar system in Joint Data. 
        '''
        gaia_datas = [data for data in self.datas if isinstance(data, GaiaData)]
        if not gaia_datas:
            raise ValueError("No Gaia data found in the joint data.")
        for system in set(data.system for data in gaia_datas):
            system_datas = [data for data in gaia_datas if data.system == system]
            spsi = np.concatenate([data.spsi for data in system_datas])
            cpsi = np.concatenate([data.cpsi for data in system_datas])
            t = np.concatenate([data.t for data in system_datas])
            plx_fac = np.concatenate([data.plx_fac for data in system_datas])
            x = np.concatenate([data.x for data in system_datas])
            err = np.concatenate([data.err for data in system_datas])

            source = system_datas[0]
            canonical_units = source.unit_system.canonical_units

            units = {"t": canonical_units["t"],
                    "x": canonical_units["x"],
                    "err": canonical_units["err"],}
            return GaiaData(spsi,cpsi,t,plx_fac,x,err, system=system,units=units)

    def as_radial_velocity_data(self):
        ''' Returns RadialVelocityData object containing all radial velocity data of similair system. 
        '''
        rv_datas = [data for data in self.datas if isinstance(data, RadialVelocityData)]
        if not rv_datas:
            raise ValueError("No radial velocity data found in the joint data.")
        for system in set(data.system for data in rv_datas):
            system_datas = [data for data in rv_datas if data.system == system]
            t = np.concatenate([data.t for data in system_datas])
            rv = np.concatenate([data.rv for data in system_datas])
            rv_err = np.concatenate([data.rv_err for data in system_datas])

            source = system_datas[0]
            canonical_units = source.unit_system.canonical_units
            units = {"t": canonical_units["t"],
                    "rv": canonical_units["rv"],
                    "rv_err": canonical_units["rv_err"],}
            return RadialVelocityData(t, rv, rv_err, system=system,units=units)


    def _concat_obs(self):
        """Returns the combined observations for all data."""
        observations = []
        for data in self.datas:
            if isinstance(data, AstrometryData):
                observations += ([data.x, data.y])
            elif isinstance(data, RadialVelocityData):
                observations.append(data.rv)
            elif isinstance(data, GaiaData):
                observations.append(data.x)
        return np.concatenate(observations)

    def _err(self):
        """Returns the combined error array for all data."""
        errors = []
        for data in self.datas:
            if isinstance(data, AstrometryData):
                errors+=([data.x_err, data.y_err])
            elif isinstance(data, RadialVelocityData):
                errors.append(data.rv_err)
            elif isinstance(data, GaiaData):
                errors.append(data.err)
        return np.concatenate(errors)


    def _astrometry(self, orbit: Orbit):
        xs, ys = [], []
        for data in self.datas:
            if isinstance(data, GaiaData) or not hasattr(data, "_astrometry"):
                continue

            try:
                x, y = data._astrometry(orbit)
            except NotImplementedError:
                continue
            xs.append(x)
            ys.append(y)
        if not xs or not ys:
            raise ValueError("No astrometry data found in the joint data.")
        return np.concatenate(xs), np.concatenate(ys)



    def _radial_velocity(self, orbit: Orbit):
        rvs = []
        for data in self.datas:
            if not hasattr(data, "_radial_velocity"):
                continue

            try:
                rv = data._radial_velocity(orbit)
            except NotImplementedError:
                continue
            rvs.append(rv)
        if not rvs:
            raise ValueError("No radial velocity data found in the joint data.")
        return np.concatenate(rvs)
    


    def t_series(self):
        ts = []
        for data in self.datas:
            t = data.t_series()
            ts.append(t)
        return np.concatenate(ts)


    def has_astrometry(self) -> bool:
            """
            Returns True if the data contains astrometry information, False otherwise.
            """
            for data in self.datas:
                if data.has_astrometry():
                    return True
            return False

    def has_radial_velocity(self) -> bool:
            """
            Returns True if the data contains radial velocity information, False otherwise.
            """
            for data in self.datas:
                if data.has_radial_velocity():
                    return True
            return False