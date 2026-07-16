# periapsis

Package for efficiently modeling and fitting orbits with various parameterizations and priors with support for data from a variety of sources.

## Installing

### pip

```bash
pip install periapsis
```

## Usage Example

```python
from periapsis.data import AstrometryData, RadialVelocityData, JointData
from periapsis.fitting import MCMCFitter
from periapsis.prior import UniformPrior
import numpy as np

fit_data = JointData([
    AstrometryData(t_astro, x, y),
    RadialVelocityData(t_rv, rv),
])

fitter = MCMCFitter(
    nwalkers=32,
    niter=10000,
    P=UniformPrior(10, 20000), # orbital period, days
    t0=UniformPrior(1990, 2050), # time of periapsis passage
    a=UniformPrior(0.01, 1000), # semi-major axis, AU
    e=UniformPrior(0, 1), # eccentricity
    cosi=UniformPrior(-1, 1), # cos(inclination)
    omega=UniformPrior(0, 2*np.pi), # argument of periapsis
    bigomega=UniformPrior(0, 2*np.pi), # longitude of ascending node
)

result = fitter.fit(fit_data)
```

