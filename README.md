# periapsis

Package for efficiently modeling and fitting orbits with various parameterizations and priors with support for data from a variety of sources.

## Installing

### pip

```bash
pip install periapsis
```

## Usage Example

```python
import periapsis as p
import numpy as np

fit_data = p.JointData([
    p.AstrometryData(t_astro, x, y,x_err,y_err,units={'t':'day','x':'mas','y':'mas','x_err':'mas','y_err':'mas'},system=1),
    p.RadialVelocityData(t_rv, rv,rv_err,units={'t':'day','rv':'km/s','rv_err':'km/s'},system=1)
])

fitter = p.MCMCFitter(
    nwalkers=32,
    niter=10000,
    P=p.UniformPrior(10, 20000), # orbital period, days
    Tp=p.UniformPrior(1990, 2050), # time of periapsis passage
    a1=p.UniformPrior(0.01, 1000), # semi-major axis, AU
    e=p.UniformPrior(0, 1), # eccentricity
    cosi=p.UniformPrior(-1, 1), # cos(inclination)
    omega1=p.UniformPrior(0, 2*np.pi), # argument of periapsis
    Omega1=p.UniformPrior(0, 2*np.pi), # longitude of ascending node
)

result = fitter.fit(fit_data)
```

