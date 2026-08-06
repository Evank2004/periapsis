from typing import Literal

all_parameters = {
    'a', 'b', 'p', 'r_a', 'r_p', 'e', 'i', 'omega', 'Omega', 'piomega', 'P', 'A', 'B', 'C', 'F', 'G', 'H', 'cosi', 'sini', 'Mtot', 'mu',
    'a1', 'b1', 'p1', 'r_a1', 'r_p1', 'omega1', 'Omega1', 'piomega1', 'A1', 'B1', 'C1', 'F1', 'G1', 'H1', 'M1',
    'a2', 'b2', 'p2', 'r_a2', 'r_p2', 'omega2', 'Omega2', 'piomega2', 'A2', 'B2', 'C2', 'F2', 'G2', 'H2', 'M2',
    'c', 'h', 'c1', 'c2', 'h1', 'h2',
    'Msini', 'M1sini', 'M2sini', 'n', 'K', 'q', 'f1', 'f2', 'minM1', 'minM2',
    'a1sini', 'a2sini',
    'Tepoch', 'Tp', 't0', 'M0', 'L0', 'E0', 'nu0', 'l0', 'uM0', 'u0',
    'u01', 'u02', 'uM01', 'uM02', 'l01', 'l02', 'K1', 'K2',
    'dalpha', 'ddelta', 'mu_alpha', 'mu_delta', 'gamma',
    'parallax', 'distance',
}

ang_parameters = {
    'a', 'b', 'p', 'r_a', 'r_p', 'A', 'B', 'C', 'F', 'G', 'H',
    'a1', 'b1', 'p1', 'r_a1', 'r_p1', 'A1', 'B1', 'C1', 'F1', 'G1', 'H1',
    'a2', 'b2', 'p2', 'r_a2', 'r_p2', 'A2', 'B2', 'C2', 'F2', 'G2', 'H2',
}

wrapped_parameters = {
    "omega", "Omega", "piomega", "omega1", "Omega1", "piomega1", "omega2", "Omega2", "piomega2",
    "t0", "M0", "L0", "E0", "nu0", "l0", "uM0", "u0", "u01", "u02", "uM01", "uM02", "l01", "l02"
}

for param in list(ang_parameters):
    all_parameters.add(f"{param}_ang")

for param in list(all_parameters):
    all_parameters.add(f"log{param}")

# TODO references and better descriptions for all parameters

a: Literal["a"] = "a"
"""Semi-major axis of relative orbit ellipse"""

a1: Literal["a1"] = "a1"
"""Semi-major axis of body 1's orbit ellipse (the orbit of body 1 with the barycenter at the focus of the ellipse)"""

a2: Literal["a2"] = "a2"
"""Semi-major axis of body 2's orbit ellipse (the orbit of body 2 with the barycenter at the focus of the ellipse)"""

b = "b"
"""Semi-minor axis of relative orbit ellipse"""

b1 = "b1"
"""Semi-minor axis of body 1's orbit ellipse (the orbit of body 1 with the barycenter at the focus of the ellipse)"""

b2 = "b2"
"""Semi-minor axis of body 2's orbit ellipse (the orbit of body 2 with the barycenter at the focus of the ellipse)"""

p = "p"
"""Semi-latus rectum (or semi-parameter) of relative orbit ellipse"""

p1 = "p1"
"""Semi-latus rectum (or semi-parameter) of body 1's orbit ellipse (the orbit of body 1 with the barycenter at the focus of the ellipse)"""

p2 = "p2"
"""Semi-latus rectum (or semi-parameter) of body 2's orbit ellipse (the orbit of body 2 with the barycenter at the focus of the ellipse)"""

r_a = "r_a"
"""Apoapsis distance to other body"""

r_a1 = "r_a1"
"""Apoapsis distance of body 1 to barycenter"""

r_a2 = "r_a2"
"""Apoapsis distance of body 2 to barycenter"""

r_p = "r_p"
"""Periapsis distance to other body. Occasionally called 'q' in the literature."""

r_p1 = "r_p1"
"""Periapsis distance of body 1 to barycenter."""

r_p2 = "r_p2"
"""Periapsis distance of body 2 to barycenter."""

e = "e"
"""Eccentricity of orbit ellipses"""

i = "i"
"""
Inclination of orbit ellipses.

`i` is measured relative to the plane of the sky, so `i`=0 is face-on, and `i`=π/2 is edge-on. 

For geocentric orbits, `i`=0 and `i`=π are equatorial and `i`=π/2 is polar. 0 < `i` < π/2 is prograde, and π/2 < `i` < π is retrograde.

For heliocentric orbits, `i`=0 and `i`=π are in the plane of the ecliptic, and `i`=π/2 is perpendicular to the ecliptic. 0 < `i` < π/2 is prograde, and π/2 < `i` < π is retrograde.
"""

cosi = "cosi"
"""
Cosine of the inclination of orbit ellipses.

For astrometric orbital solutions, only `cosi` is constrained, and not `i` itself.
"""

sini = "sini"
"""Sine of the inclination of orbit ellipses."""

omega = "omega"
"""
Argument of periastron.

`ω` (omega) measures the angle in the orbital plane from the ascending node to the periapsis of the orbit of body 2 (the secondary).
"""

omega1 = "omega1"
"""
Argument of periastron of body 1. Related to `omega` by `omega1 = omega + π`.
"""

omega2 = "omega2"
"""
Argument of periastron of body 2. Equivalent to `omega` by convention.
"""

Omega = "Omega"
"""
Longitude of ascending node.

`Ω` (Omega) measures the angle in the reference plane from the reference direction to the ascending node of the orbit of body 2 (the secondary).

If the reference plane is the plane of the sky and the reference direction is North.

For geocentric orbits, the reference plane is the equatorial plane and the reference direction is the vernal equinox.

For heliocentric orbits, the reference plane is the ecliptic plane and the reference direction is the first point of Aries (the vernal equinox).
"""
# TODO - verify reference directions in our code. Also need to distinguish between longitude of ascending node (0 < Ω < 2π) and longitude of node (0 < Ω < π) when only absolute inclination is known. 

Omega1 = "Omega1"
"""
Longitude of ascending node of body 1. Related to `Omega` by `Omega1 = Omega + π`.
"""

Omega2 = "Omega2"
"""
Longitude of ascending node of body 2. Equivalent to `Omega` by convention.
"""

piomega = "piomega"
"""
Longitude of periastron.

`ϖ` (piomega) measures the angle in the reference plane from the reference direction to the periapsis of the orbit of body 2 (the secondary). It is related to `ω` and `Ω` by `ϖ = ω + Ω`.

If the reference plane is the plane of the sky and the reference direction is North.

For geocentric orbits, the reference plane is the equatorial plane and the reference direction is the vernal equinox.

For heliocentric orbits, the reference plane is the ecliptic plane and the reference direction is the first point of Aries (the vernal equinox).
"""

piomega1 = "piomega1"
"""
Longitude of periastron of body 1. Related to `piomega` by `piomega1 = piomega + π`.
"""

piomega2 = "piomega2"
"""
Longitude of periastron of body 2. Equivalent to `piomega` by convention.
"""

P = "P"
"""Orbital period"""

n = "n"
"""
Mean motion - angle per unit time.

Defined as `n = 2π/P`, where `P` is the orbital period. The mean motion is the average angular speed required for a body to complete one orbit.
"""

A = "A"
"""Thiele-Innes constant A for relative ellipse."""

A1 = "A1"
"""Thiele-Innes constant A for orbital ellipse of body 1."""

A2 = "A2"
"""Thiele-Innes constant A for orbital ellipse of body 2."""

B = "B"
"""Thiele-Innes constant B for relative ellipse."""

B1 = "B1"
"""Thiele-Innes constant B for orbital ellipse of body 1."""

B2 = "B2"
"""Thiele-Innes constant B for orbital ellipse of body 2."""

C = "C"
"""Thiele-Innes constant C for relative ellipse."""

C1 = "C1"
"""Thiele-Innes constant C for orbital ellipse of body 1."""

C2 = "C2"
"""Thiele-Innes constant C for orbital ellipse of body 2."""

F = "F"
"""Thiele-Innes constant F for relative ellipse."""

F1 = "F1"
"""Thiele-Innes constant F for orbital ellipse of body 1."""

F2 = "F2"
"""Thiele-Innes constant F for orbital ellipse of body 2."""

G = "G"
"""Thiele-Innes constant G for relative ellipse."""

G1 = "G1"
"""Thiele-Innes constant G for orbital ellipse of body 1."""

G2 = "G2"
"""Thiele-Innes constant G for orbital ellipse of body 2."""

H = "H"
"""Thiele-Innes constant H for relative ellipse."""

H1 = "H1"
"""Thiele-Innes constant H for orbital ellipse of body 1."""

H2 = "H2"
"""Thiele-Innes constant H for orbital ellipse of body 2."""

c = "c"
"""Thiele-Innes velocity constant c for relative ellipse."""

c1 = "c1"
"""Thiele-Innes velocity constant c for orbital ellipse of body 1."""

c2 = "c2"
"""Thiele-Innes velocity constant c for orbital ellipse of body 2."""

h = "h"
"""Thiele-Innes velocity constant h for relative ellipse."""

h1 = "h1"
"""Thiele-Innes velocity constant h for orbital ellipse of body 1."""

h2 = "h2"
"""Thiele-Innes velocity constant h for orbital ellipse of body 2."""

Mtot = "Mtot"
"""Total mass of the system."""

M1 = "M1"
"""Mass of body 1."""

M2 = "M2"
"""Mass of body 2."""

minM1 = "minM1"
"""Minimum mass of body 1, assuming sin(i) = 1."""

minM2 = "minM2"
"""Minimum mass of body 2, assuming sin(i) = 1."""

Msini = "Msini"
"""Mtot * sini"""

mu = "mu"
"""Mtot * G, where G is the gravitational constant."""

Tepoch = "Tepoch"
"""Reference epoch for orbital elements."""

Tp = "Tp"
"""
Time of periastron passage.

This is the absolute time at which the orbiting bodies pass through periapsis.
"""

t0 = "t0"
"""Scaled time of periastron passage (t0 = (Tp - Tepoch) mod P)."""

M0 = "M0"
"""
Mean anomaly at reference epoch.

Mean anomaly is the angle measured in the orbital plane from periapsis to the position of the body if it were moving at a constant angular speed throughout its whole orbit.
"""

L0 = "L0"
"""
Mean longitude at reference epoch.

Mean longitude is the angle measured in the reference plane from the reference direction to the position of the body if it were moving at a constant angular speed throughout its whole orbit.
"""
# TODO check this one. Clean up anomaly/longitude descriptions

E0 = "E0"
"""
Eccentric anomaly at reference epoch.

Eccentric anomaly is the angle measured in the auxiliary circle of the ellipse from the center of the ellipse to the position of the body if it were moving at a constant angular speed throughout its whole orbit.
"""

nu0 = "nu0"
"""
True anomaly at reference epoch.

True anomaly is the angle measured in the orbital plane from periapsis to the position of the body relative to the focus of the ellipse.
"""

l0 = "l0"
"""
True longitude at reference epoch.

True longitude is the angle measured in the reference plane from the reference direction to the position of the body relative to the focus of the ellipse.
"""

l01 = "l01"
"""
True longitude of body 1 at reference epoch.

True longitude is the angle measured in the reference plane from the reference direction to the position of the body relative to the focus of the ellipse.
This is the true longitude of body 1, which is related to the true longitude by l01 = l0 + π.
"""

l02 = "l02"
"""
True longitude of body 2 at reference epoch.
True longitude is the angle measured in the reference plane from the reference direction to the position of the body relative to the focus of the ellipse.
This is the true longitude of body 2, which is related to the true longitude of body 1 by l02 = l01 + π.
"""

uM0 = "uM0"
"""
Mean argument of latitude at reference epoch.

Mean argument of latitude is the angle measured in the orbital plane from the ascending node to the position of the body if it were moving at a constant angular speed throughout its whole orbit.
"""

uM01 = "uM01"
"""
Mean argument of latitude of body 1 at reference epoch.

Mean argument of latitude is the angle measured in the orbital plane from the ascending node to the position of the body if it were moving at a constant angular speed throughout its whole orbit.
This is the mean argument of latitude of body 1, which is related to the mean argument of latitude by uM01 = uM0 + π.
"""

uM02 = "uM02"
"""
Mean argument of latitude of body 2 at reference epoch.
Mean argument of latitude is the angle measured in the orbital plane from the ascending node to the position of the body if it were moving at a constant angular speed throughout its whole orbit.
This is the mean argument of latitude of body 2, which is related to the mean argument of latitude of body 1 by uM02 = uM01 + π.
"""

u0 = "u0"
"""
True argument of latitude at reference epoch.

True argument of latitude is the angle measured in the orbital plane from the ascending node to the position of the body relative to the focus of the ellipse.
"""

u01 = "u01"
"""
True argument of latitude of body 1 at reference epoch.

True argument of latitude is the angle measured in the orbital plane from the ascending node to the position of the body relative to the focus of the ellipse.
This is the true argument of latitude of body 1, which is related to the true argument of latitude by u01 = u0 + π.
"""

u02 = "u02"
"""
True argument of latitude of body 2 at reference epoch.


True argument of latitude is the angle measured in the orbital plane from the ascending node to the position of the body relative to the focus of the ellipse.
This is the true argument of latitude of body 2, which is related to the true argument of latitude of body 1 by u02 = u01 + π.
"""

K = "K"
"""
Relative radial velocity semi-amplitude.

Most likely you want to use K1 or K2 instead, since K is not directly observable.
"""

K1 = "K1"
"""
Radial velocity semi-amplitude of the body 1 (the primary).
"""

K2 = "K2"
"""
Radial velocity semi-amplitude of the body 2 (the secondary).
"""

q = "q"
"""
Mass ratio of the system.

We define q = M2/M1

There are varying conventions. Usually M1 is the more massive body. In some contexts, M1 is the brighter (but not necessarily more massive) body.
"""

dalpha = "dalpha"
"""
Astrometric offset of the barycenter at reference epoch in ΔRA*cos(Dec) from the origin of the on-sky tangential projection.
"""

ddelta = "ddelta"
"""
Astrometric offset of the barycenter at reference epoch in Dec from the origin of the on-sky tangential projection.
"""

mu_alpha = "mu_alpha"
"""
Proper motion of the barycenter measured in ΔRA*cos(Dec) direction.
"""

mu_delta = "mu_delta"
"""
Proper motion of the barycenter measured in Dec direction.
"""

gamma = "gamma"
"""
Systemic radial velocity of the barycenter of the system.
"""

f1 = "f1"
"""
Mass function relative to body 1.

The mass function is a quantity derived from observations that constrains the masses of the two bodies in a binary system.

Although f1 is mainly used to constrain the mass of the body 2, it is derived from the radial velocity semi-amplitude of body 1 (K1), and is therefore called the mass function of body 1.

f1 = (M2^3 * sin(i)^3) / (M1 + M2)^2 = (K1^3 * P) / (2πG) * (1-e^2)^(3/2)
"""

f2 = "f2"
"""
Mass function relative to body 2.

The mass function is a quantity derived from observations that constrains the masses of the two bodies in a binary system.

Although f2 is mainly used to constrain the mass of the body 1, it is derived from the radial velocity semi-amplitude of body 2 (K2), and is therefore called the mass function of body 2.

f2 = (M1^3 * sin(i)^3) / (M1 + M2)^2 = (K2^3 * P) / (2πG) * (1-e^2)^(3/2)
"""

parallax = "parallax"
"""
Parallax of the system.

Objects outside the solar system trace an ellipse on the sky as an observer orbits the Sun. The parallax is the angular size of the semi-major axis of this ellipse.

The details of the parallax ellipse depend on the position of the object in the sky, the observer's orbit around the Sun, and the distance from the object to the observer.

This only forms a true ellipse for objects outside the solar system which have a distance much greater than the size of the observer's orbit around the Sun.
"""

distance = "distance"
"""
Distance from the observer to the system.

This is only well-defined for objects outside the solar system which have a distance much greater than the size of the observer's orbit around the Sun. At this distance, the difference between helio- and geo-centric distances is negligible.
"""


def log(param):
    """
    Use the log10 scaled version of a parameter. For example, log(a) is the log10 of the semi-major axis.

    This can be useful for parameters that span many orders of magnitude, such as masses and distances.
    """
    return f"log{param}"


def ang(param):
    """
    Use the angular scaled version of a parameter. For example, ang(a) is the semi-major axis in angular units (e.g. milliarcseconds).

    This can be useful for fitting systems without a known distance/parallax, or where the distance/parallax is a free parameter.
    """
    if param not in ang_parameters:
        raise ValueError(f"Parameter {param} does not have an angular version.")
    return f"{param}_ang"


_display_dict = {
    'r_a': 'rₐ',
    'r_p': 'rₚ',
    'omega': 'ω',
    'Omega': 'Ω',
    'piomega': 'ϖ',
    'mu': 'μ',
    'Tp': 'Tₚ',
    't0': 't₀',
    'M0': 'M₀',
    'L0': 'L₀',
    'E0': 'E₀',
    'nu0': 'ν₀',
    'l0': 'l₀',
    'uM0': 'u_M₀',
    'u0': 'u₀',
    'dalpha': 'Δα',
    'ddelta': 'Δδ',
    'mu_alpha': 'μ_α',
    'mu_delta': 'μ_δ',
    'gamma': 'γ',
}


def display(param):
    """
    Get a display-friendly version of a parameter name.
    """
    if param.startswith("log"):
        return f"log₁₀({display(param[3:])})"
    elif param.endswith("_ang"):
        return f"{display(param[:-4])} (angular)"

    if param.endswith("1"):
        return f"{display(param[:-1])}₁"
    elif param.endswith("2"):
        return f"{display(param[:-1])}₂"
    else:
        return _display_dict[param] if param in _display_dict else param





