import numpy as np
from periapsis.utils.solvers import transform_theile
from periapsis.utils.solvers import solve_mass


class FitResults:
    def __init__(self, **samples):
        self.raw_samples = samples.pop('raw_sampler', None)
        self.backend = samples.pop('backend', None)
        self.fit_method = samples.pop('fit_method', None)
        self.MAP_params = samples.pop('MAP_params', None)
        self.median_params = samples.pop('median_params', None)
        self.PM_fit = samples.pop('PM_fit', None)
        self.param_names = samples.pop('param_names', None)
        self.sampler = self.backend
        self.m1 = samples.pop('m1', None)

        self.samples = samples
        if self.param_names is not None:
            self.samples.setdefault('param_names', self.param_names)

    def add_mass_samples(self, m1=None):
        """Add secondary-mass samples derived from the orbital samples."""
        if m1 is None:
            m1 = self.m1
        if m1 is None:
            return

        param_names = self.param_names or self.samples.get('param_names', [])
        if not param_names:
            return

        period_name = next((name for name in param_names if name in {'P', 'p', 'period', 'Period'}), None)
        a1_name = next((name for name in param_names if name in {'a1', 'a', 'semimajoraxis', 'semi_major_axis'}), None)

        if period_name is None:
            return

        if a1_name is None:
            # Attempt to compute a1 from Thiele-Innes parameters if available
            A_name = next((name for name in param_names if name == 'A'), None)
            B_name = next((name for name in param_names if name == 'B'), None)
            F_name = next((name for name in param_names if name == 'F'), None)
            G_name = next((name for name in param_names if name == 'G'), None)

            if A_name and B_name and F_name and G_name:
                A_samps = self.samples.get(A_name)
                B_samps = self.samples.get(B_name)
                F_samps = self.samples.get(F_name)
                G_samps = self.samples.get(G_name)

                a1_samps, _, _, _ = transform_theile(A_samps, B_samps, F_samps, G_samps)
                self.samples['a1'] = a1_samps
                a1_name = 'a1'
            else:
                return

        P_samps = self.samples.get(period_name)
        a1_samps = self.samples.get(a1_name)
        

        m2_samps = solve_mass(np.asarray(a1_samps, dtype=float), np.asarray(P_samps, dtype=float), float(m1))
        m2_samps = np.where(np.isfinite(m2_samps) & (m2_samps > 0), m2_samps, np.nan)
        self.samples['M2'] = m2_samps
        setattr(self,'M2',m2_samps)