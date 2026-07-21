import numpy as np
from periapsis.utils.solvers import solve_mass, solve_kepler

from collections import defaultdict, deque
from functools import lru_cache
from typing import Callable

# a = semi-major axis
# b = semi-minor axis
# p = semi-parameter / semi-latus rectum
# r_a = apoapsis distance
# r_p = periapsis distance (sometimes called q)
# e = eccentricity
# i = inclination
# omega = argument of periastron
# Omega = longitude of ascending node
# piomega = longitude of periastron (omega + Omega)
# P = orbital period
# A,
# B, 
# F, 
# G = Thiele-Innes constants
# cosi = cos(i)
# sini = sin(i)
# Mtot = total mass of the system
# a1 = semi-major axis of the primary
# a2 = semi-major axis of the secondary
# M1 = mass of the primary
# M2 = mass of the secondary
# Msini = Mtot * sini
# n = mean motion - angle per unit time
# mu = Mtot*G
# Tepoch = reference epoch
# Tp = time of periastron passage
# t0 = Scaled time of periastron passage (t0 = (Tp - Tepoch)/P)
# M0 = mean anomaly at reference epoch
# L0 = mean longitude at reference epoch (longitude measured wrt vernal point)
# E0 = eccentric anomaly at reference epoch
# nu0 = true anomaly at reference epoch (sometimes called theta0)
# l0 = true longitude at reference epoch
# uM0 = mean argument of latitude at reference epoch (latitude measured wrt ascending node)
# u0 = true argument of latitude at reference epoch
# K = radial velocity semi-amplitude
# K1 = radial velocity semi-amplitude of the primary
# K2 = radial velocity semi-amplitude of the secondary
# q = mass ratio = M2/M1
# dx = astrometric offset in x
# dy = astrometric offset in y
# dpmra = proper motion offset in RA
# dpmdec = proper motion offset in Dec
# systemic_velocity = systemic velocity offset in radial velocity

# TODO add log versions of some params
# TODO add T-I C and H
# TODO add equinoctal params
# TODO add vector params
# TODO add eclipsing binary/planet params
# TODO add Delauny variables
# TODO add quaternion params
# TODO add flux of source 1/2 (and scale stuff in orbit by that - add photocenter option in addition to 1,2,relative)

_all_parameters = {
    'a', 'b', 'p', 'r_a', 'r_p', 'e', 'i', 'omega', 'Omega', 'piomega', 'P', 'A', 'B', 'F', 'G', 'cosi', 'sini', 'Mtot', 'mu',
    'a1', 'b1', 'p1', 'r_a1', 'r_p1', 'omega1', 'piomega1', 'A1', 'B1', 'F1', 'G1', 'M1',
    'a2', 'b2', 'p2', 'r_a2', 'r_p2', 'omega2', 'piomega2', 'A2', 'B2', 'F2', 'G2', 'M2',
    'Msini', 'M1sini', 'M2sini', 'n', 'K', 'q',
    'a1sini', 'a2sini',
    'Tepoch', 'Tp', 't0', 'M0', 'L0', 'E0', 'nu0', 'l0', 'uM0', 'u0',
    'u01', 'u02', 'uM01', 'uM02', 'l01', 'l02', 'K1', 'K2',
    'dx', 'dy', 'dpmra', 'dpmdec', 'systemic_velocity'
}

def A_B_F_G_to_a_cosi_omega_Omega(A, B, F, G):
    popovic_k = (A*A + B*B + F*F + G*G) / 2.0
    popovic_m = A*G - B*F
    popovic_j = np.sqrt(np.maximum(0, popovic_k * popovic_k - popovic_m*popovic_m))
    a_squared = popovic_j + popovic_k
    a1 = np.sqrt(a_squared) # as
    cosi = popovic_m / a_squared
    cosi = np.where(a_squared == 0, 1.0, cosi)
    # i = np.arctan2(a1 * np.sqrt(2.0 * popovic_j), popovic_m) # quadrant issue. This should have two solutions. For now just returning cosi
    vpi = np.arctan2(B-F, A+G)
    Omo = np.arctan2(B+F, A-G)
    long = np.mod((vpi + Omo) / 2.0 + 2*np.pi, np.pi)
    w = np.mod(vpi - long + 2*np.pi, 2*np.pi) 
    return a1, cosi, w, long

def a_cosi_omega_Omega_to_ABFG(a, cosi, omega, Omega):
    cO = np.cos(Omega)
    sO = np.sin(Omega)
    cw = np.cos(omega)
    sw = np.sin(omega)

    A = a * (cO * cw - sO * sw * cosi)
    B = a * (sO * cw + cO * sw * cosi)
    F = a * (-cO * sw - sO * cw * cosi)
    G = a * (-sO * sw + cO * cw * cosi)

    return A, B, F, G

def P_a_to_Mtot(P, a):
    # TODO units
    return a**3 / P**2

def Mtot_a_to_P(Mtot, a):
    # TODO units
    return np.sqrt(a**3 / Mtot)

def Mtot_P_to_a(Mtot, P):
    # TODO units
    return np.cbrt(Mtot * P**2)

def Ma_aa_P_to_Mb(Ma, aa, P):
    # a_1**3 / P**2 = M2**3 / (M1 + M2)**2 (ignoring units...)
    Mb = solve_mass(aa, P, Ma) # TODO: migrate solver to this file and make private?
    return Mb

def aa_Ma_Mb_to_ab_a(aa, Ma, Mb):
    ab = aa * Ma/Mb
    a = aa + ab
    return ab, a

def a_M1_M2_to_a1_a2(a, M1, M2):
    a1 = a * M2 / (M1 + M2)
    a2 = a * M1 / (M1 + M2)
    return a1, a2

def omega_to_omega2_omega1(omega):
    omega2 = omega
    omega1 = np.mod(omega + np.pi, 2 * np.pi)
    return omega2, omega1

def omega1_to_omega_omega2(omega1):
    omega = np.mod(omega1 + np.pi, 2 * np.pi)
    omega2 = omega
    return omega, omega2

def omega2_to_omega_omega1(omega2):
    omega = omega2
    omega1 = np.mod(omega2 + np.pi, 2 * np.pi)
    return omega, omega1

def a_b_to_e(a, b):
    e = np.where(a > 0, np.sqrt(1 - (b/a)**2), np.where(a < 0, np.sqrt(1 + (b/a)**2), np.nan))
    return e

def i_to_cosi_sini(i):
    return np.cos(i), np.sin(i)

def P_to_n(P):
    return 2 * np.pi / P

def n_to_P(n):
    return 2 * np.pi / n

def a_e_to_b_p_rp_ra(a, e):
    one_minus_e2 = 1 - e**2
    b = a * np.sqrt(np.abs(one_minus_e2))
    p = a * one_minus_e2
    r_p = a * (1 - e)
    r_a = a * (1 + e)
    return b, p, r_p, r_a

def b_e_to_a(b, e):
    a = b / np.sqrt(np.abs(1-e**2))
    return a

def ra_rp_to_e_a(r_a, r_p):
    e = (r_a - r_p) / (r_a + r_p)
    a = (r_a + r_p) / 2
    return e, a

def rp_e_to_ra_a(r_p, e):
    r_a = r_p * (1 + e) / (1 - e)
    a = (r_a + r_p) / 2
    return r_a, a

def ra_e_to_rp_a(r_a, e):
    r_p = r_a * (1 - e) / (1 + e)
    a = (r_a + r_p) / 2
    return r_p, a

def ra_a_to_rp_e(r_a, a):
    r_p = 2*a - r_a
    e = (r_a - r_p) / (r_a + r_p)
    return r_p, e

def rp_a_to_ra_e(r_p, a):
    r_a = 2*a - r_p
    e = (r_a - r_p) / (r_a + r_p)
    return r_a, e

def p_e_to_a(p, e):
    a = p/(1 - e**2)
    return a

def a_p_to_e(a, p):
    e = np.sqrt(1 - p/a)
    return e

def Mtot_to_mu(Mtot):
    # TODO: units
    return Mtot * 4*np.pi**2

def mu_to_Mtot(mu):
    # TODO: units
    return mu / (4*np.pi**2)

def mu_a_to_n(mu, a):
    n = np.sqrt(mu / np.abs(a)**3)
    return n

def n_a_to_mu(n, a):
    mu = n**2 * np.abs(a)**3
    return mu

def mu_n_to_a(mu, n):
    a = np.cbrt(mu / n**2)
    return a

def Tp_Tepoch_P_to_t0(Tp, Tepoch, P):
    t0 = (Tp - Tepoch) / P
    return t0

def t0_Tp_Tepoch_to_P(t0, Tp, Tepoch):
    P = (Tp - Tepoch) / t0
    return P

def P_t0_Tp_to_Tepoch(P, t0, Tp):
    Tepoch = Tp - t0 * P
    return Tepoch

def Tepoch_P_t0_to_Tp(Tepoch, P, t0):
    Tp = Tepoch + t0 * P
    return Tp

def Tepoch_Tp_M0_to_n(Tepoch, Tp, M0):
    n = M0 / (Tepoch - Tp)
    return n

def Tepoch_M0_n_to_Tp(Tepoch, M0, n):
    Tp = Tepoch - M0 / n
    return Tp

def Tepoch_n_Tp_to_M0(Tepoch, n, Tp):
    M0 = n * (Tepoch - Tp)
    return M0

def M0_E0_to_e(M0, E0):
    e = (E0 - M0) / np.sin(E0) # TODO divide by zero issue?
    return e

def E0_e_to_M0_nu0(E0, e):
    M0 = E0 - e * np.sin(E0)
    nu0 = np.arctan2(np.sqrt(1 - e**2) * np.sin(E0), np.cos(E0) - e)
    return M0, nu0

def e_M0_to_E0(e, M0):
    # M0 = E0 - e * sin(E0)
    E0 = solve_kepler(M0, e)
    return E0

def nu0_e_to_E0(nu0, e):
    E0 = np.arctan2(np.sin(nu0) * np.sqrt(1 - e**2), e + np.cos(nu0))
    return E0

def u0_to_u01_u02(u0):
    u01 = np.mod(u0 + np.pi, 2*np.pi)
    u02 = u0
    return u01, u02

def u01_to_u0_u02(u01):
    u0 = np.mod(u01 + np.pi, 2*np.pi)
    u02 = u0
    return u0, u02

def u02_to_u0_u01(u02):
    u0 = u02
    u01 = np.mod(u0 + np.pi, 2*np.pi)
    return u0, u01

def uM0_to_uM01_uM02(uM0):
    uM01 = np.mod(uM0 + np.pi, 2*np.pi)
    uM02 = uM0
    return uM01, uM02

def uM01_to_uM0_uM02(uM01):
    uM0 = np.mod(uM01 + np.pi, 2*np.pi)
    uM02 = uM0
    return uM0, uM02

def uM02_to_uM0_uM01(uM02):
    uM0 = uM02
    uM01 = np.mod(uM0 + np.pi, 2*np.pi)
    return uM0, uM01

def a_sini_n_e_to_K(a, sini, n, e):
    # TODO: These functions return K in units of length/time that match the units of asini and n. Typically AU/yr. Need to better handle conversion factors.
    K = n*a*sini/np.sqrt(1 - e**2)
    return K

def K_n_e_a_to_sini(K, n, e, a):
    sini = K*np.sqrt(1 - e**2)/(n*a)
    return sini

def K_e_a_sini_to_n(K, e, a, sini):
    n = K*np.sqrt(1 - e**2)/(a*sini)
    return n

def K_a_sini_n_to_e(K, a, sini, n):
    e = np.sqrt(1 - ((n*a*sini)/K)**2)
    return e

def K_sini_n_e_to_a(K, sini, n, e):
    a = K*np.sqrt(1-e**2)/(n*sini)
    return a

def Mtot_q_to_M1_M2(Mtot, q):
    M1 = Mtot / (1 + q)
    M2 = Mtot * q / (1 + q)
    return M1, M2

def n_K_e_to_asini(n, K, e):
    asini = K * np.sqrt(1-e**2) / n
    return asini

def asini_n_e_to_K(asini, n, e):
    # TODO: These functions return K in units of length/time that match the units of asini and n. Typically AU/yr. Need to better handle conversion factors.
    K = n * asini / np.sqrt(1-e**2)
    return K

def K_e_asini_to_n(K, e, asini):
    n = K * np.sqrt(1-e**2) / asini
    return n

def n_asini_K_to_e(n, asini, K):
    e = np.sqrt(1-(n*asini/K)**2)
    return e

def Ma_aasini_Mbsini_to_ab(Ma, aasini, Mbsini):
    ab = Ma * aasini / Mbsini
    return ab

def aasini_Mbsini_ab_to_Ma(aasini, Mbsini, ab):
    Ma = Mbsini * ab / aasini
    return Ma

def Mbsini_ab_Ma_to_aasini(Mbsini, ab, Ma):
    aasini = Mbsini * ab / Ma
    return aasini

def ab_Ma_aasini_to_Mbsini(ab, Ma, aasini):
    Mbsini = Ma * aasini / ab
    return Mbsini

def add_ab(a, b):
    return a + b

def add_ab_mod2pi(a, b):
    return np.mod(a + b, 2*np.pi)

def sub_ab(a, b):
    return a - b

def sub_ab_mod2pi(a, b):
    return np.mod(a - b, 2*np.pi)

def mul_ab(a, b):
    return a * b

def div_ab(a, b):
    return a / b

def identity(a):
    return a




_transform_graph = [
    (('A', 'B', 'F', 'G',), ('a', 'cosi', 'omega', 'Omega',), A_B_F_G_to_a_cosi_omega_Omega),
    (('A1', 'B1', 'F1', 'G1',), ('a1', 'cosi', 'omega1', 'Omega',), A_B_F_G_to_a_cosi_omega_Omega),
    (('A2', 'B2', 'F2', 'G2',), ('a2', 'cosi', 'omega2', 'Omega',), A_B_F_G_to_a_cosi_omega_Omega),
    (('a', 'cosi', 'omega', 'Omega',), ('A', 'B', 'F', 'G',), a_cosi_omega_Omega_to_ABFG),
    (('a1', 'cosi', 'omega1', 'Omega',), ('A1', 'B1', 'F1', 'G1',), a_cosi_omega_Omega_to_ABFG),
    (('a2', 'cosi', 'omega2', 'Omega',), ('A2', 'B2', 'F2', 'G2',), a_cosi_omega_Omega_to_ABFG),
    (('P', 'a',), ('Mtot',), P_a_to_Mtot),
    (('Mtot', 'a',), ('P',), Mtot_a_to_P),
    (('Mtot', 'P',), ('a',), Mtot_P_to_a),
    (('Mtot', 'M1',), ('M2',), sub_ab),
    (('Mtot', 'M2',), ('M1',), sub_ab),
    (('Mtot', 'sini',), ('Msini',), mul_ab),
    (('Msini', 'Mtot',), ('sini',), div_ab),
    (('Msini', 'sini',), ('Mtot',), div_ab),
    (('M1', 'sini',), ('M1sini',), mul_ab),
    (('M1sini', 'M1',), ('sini',), div_ab),
    (('M1sini', 'sini',), ('M1',), div_ab),
    (('M2', 'sini',), ('M2sini',), mul_ab),
    (('M2sini', 'M2',), ('sini',), div_ab),
    (('M2sini', 'sini',), ('M2',), div_ab),
    (('sini', 'a1'), ('a1sini',), mul_ab),
    (('sini', 'a2'), ('a2sini',), mul_ab),
    (('a1sini', 'a1'), ('sini',), div_ab),
    (('a2sini', 'a2'), ('sini',), div_ab),
    (('a1sini', 'sini'), ('a1',), div_ab),
    (('a2sini', 'sini'), ('a2',), div_ab),
    (('a1', 'a2',), ('a',), add_ab),
    (('M1', 'M2',), ('Mtot',), add_ab),
    (('M2', 'a2', 'P'), ('M1',), Ma_aa_P_to_Mb),
    (('M1', 'a1', 'P'), ('M2',), Ma_aa_P_to_Mb),
    (('a1', 'M1', 'M2',), ('a2', 'a'), aa_Ma_Mb_to_ab_a),
    (('a2', 'M2', 'M1',), ('a1', 'a'), aa_Ma_Mb_to_ab_a),
    (('a', 'M1', 'M2',), ('a1', 'a2'), a_M1_M2_to_a1_a2),
    (('omega', 'Omega',), ('piomega',), add_ab_mod2pi), # TODO may need some modulo here
    (('omega1', 'Omega',), ('piomega1',), add_ab_mod2pi),
    (('omega2', 'Omega',), ('piomega2',), add_ab_mod2pi),
    (('omega',), ('omega2', 'omega1',), omega_to_omega2_omega1),
    (('omega1',), ('omega', 'omega2',), omega1_to_omega_omega2),
    (('omega2',), ('omega', 'omega1',), omega2_to_omega_omega1),
    (('piomega', 'Omega',), ('omega',), sub_ab_mod2pi),
    (('piomega1', 'Omega',), ('omega1',), sub_ab_mod2pi),
    (('piomega2', 'Omega',), ('omega2',), sub_ab_mod2pi),
    (('piomega', 'omega',), ('Omega',), sub_ab_mod2pi),
    (('piomega1', 'omega1',), ('Omega',), sub_ab_mod2pi),
    (('piomega2', 'omega2',), ('Omega',), sub_ab_mod2pi),
    (('i',), ('cosi', 'sini',), i_to_cosi_sini),
    (('a', 'b',), ('e',), a_b_to_e),
    (('a1', 'b1',), ('e',), a_b_to_e),
    (('a2', 'b2',), ('e',), a_b_to_e),
    (('a', 'e',), ('b', 'p', 'r_p', 'r_a',), a_e_to_b_p_rp_ra),
    (('a1', 'e',), ('b1', 'p1', 'r_p1', 'r_a1',), a_e_to_b_p_rp_ra),
    (('a2', 'e',), ('b2', 'p2', 'r_p2', 'r_a2',), a_e_to_b_p_rp_ra),
    (('b', 'e',), ('a',), b_e_to_a),
    (('b1', 'e',), ('a1',), b_e_to_a),
    (('b2', 'e',), ('a2',), b_e_to_a),
    (('r_a', 'r_p',), ('e', 'a'), ra_rp_to_e_a),
    (('r_a1', 'r_p1',), ('e', 'a1'), ra_rp_to_e_a),
    (('r_a2', 'r_p2',), ('e', 'a2'), ra_rp_to_e_a),
    (('r_p', 'e',), ('r_a', 'a'), rp_e_to_ra_a),
    (('r_p1', 'e',), ('r_a1', 'a1'), rp_e_to_ra_a),
    (('r_p2', 'e',), ('r_a2', 'a2'), rp_e_to_ra_a),
    (('r_a', 'e',), ('r_p', 'a'), ra_e_to_rp_a),
    (('r_a1', 'e',), ('r_p1', 'a1'), ra_e_to_rp_a),
    (('r_a2', 'e',), ('r_p2', 'a2'), ra_e_to_rp_a),
    (('r_a', 'a',), ('r_p', 'e'), ra_a_to_rp_e),
    (('r_a1', 'a1',), ('r_p1', 'e'), ra_a_to_rp_e),
    (('r_a2', 'a2',), ('r_p2', 'e'), ra_a_to_rp_e),
    (('r_p', 'a',), ('r_a', 'e'), rp_a_to_ra_e),
    (('r_p1', 'a1',), ('r_a1', 'e'), rp_a_to_ra_e),
    (('r_p2', 'a2',), ('r_a2', 'e'), rp_a_to_ra_e),
    (('p', 'e',), ('a',), p_e_to_a),
    (('p1', 'e',), ('a1',), p_e_to_a),
    (('p2', 'e',), ('a2',), p_e_to_a),
    (('a', 'p',), ('e',), a_p_to_e),
    (('a1', 'p1',), ('e',), a_p_to_e),
    (('a2', 'p2',), ('e',), a_p_to_e),
    (('Mtot',), ('mu',), Mtot_to_mu),
    (('mu',), ('Mtot',), mu_to_Mtot),
    (('P',), ('n',), P_to_n),
    (('n',), ('P',), n_to_P),
    (('mu', 'a',), ('n',), mu_a_to_n),
    (('n', 'a',), ('mu',), n_a_to_mu),
    (('mu', 'n',), ('a',), mu_n_to_a),
    # (('Tp',), ('Tepoch',), identity),
    (('Tp', 'Tepoch', 'P',), ('t0',), Tp_Tepoch_P_to_t0),
    (('t0', 'Tp', 'Tepoch',), ('P',), t0_Tp_Tepoch_to_P),
    (('P', 't0', 'Tp',), ('Tepoch',), P_t0_Tp_to_Tepoch),
    (('Tepoch', 'P', 't0',), ('Tp',), Tepoch_P_t0_to_Tp),
    (('Tepoch', 'Tp', 'M0',), ('n',), Tepoch_Tp_M0_to_n),
    (('Tepoch', 'M0', 'n',), ('Tp',), Tepoch_M0_n_to_Tp),
    (('Tepoch', 'n', 'Tp',), ('M0',), Tepoch_n_Tp_to_M0),
    (('M0', 'E0',), ('e',), M0_E0_to_e),
    (('E0', 'e',), ('M0', 'nu0',), E0_e_to_M0_nu0),
    (('e', 'M0',), ('E0',), e_M0_to_E0),
    (('L0', 'M0',), ('piomega',), sub_ab_mod2pi), # TODO may need some modulo here
    (('piomega', 'M0',), ('L0',), add_ab_mod2pi),
    (('L0', 'piomega',), ('M0',), sub_ab_mod2pi),
    (('l0', 'nu0',), ('piomega',), sub_ab_mod2pi),
    (('l01', 'nu0',), ('piomega1',), sub_ab_mod2pi),
    (('l02', 'nu0',), ('piomega2',), sub_ab_mod2pi),
    (('piomega', 'nu0',), ('l0',), add_ab_mod2pi),
    (('piomega1', 'nu0',), ('l01',), add_ab_mod2pi),
    (('piomega2', 'nu0',), ('l02',), add_ab_mod2pi),
    (('l0', 'piomega',), ('nu0',), sub_ab_mod2pi),
    (('l01', 'piomega1',), ('nu0',), sub_ab_mod2pi),
    (('l02', 'piomega2',), ('nu0',), sub_ab_mod2pi),
    (('uM0', 'M0',), ('Omega',), sub_ab_mod2pi),
    (('Omega', 'M0',), ('uM0',), add_ab_mod2pi),
    (('uM0', 'Omega',), ('M0',), sub_ab_mod2pi),
    (('u0', 'nu0',), ('Omega',), sub_ab_mod2pi),
    (('Omega', 'nu0',), ('u0',), add_ab_mod2pi),
    (('u0', 'Omega',), ('nu0',), sub_ab_mod2pi),
    (('nu0', 'e',), ('E0',), nu0_e_to_E0),
    (('u0',), ('u01', 'u02'), u0_to_u01_u02),
    (('u01',), ('u0', 'u02'), u01_to_u0_u02),
    (('u02',), ('u0', 'u01'), u02_to_u0_u01),
    (('uM0',), ('uM01', 'uM02'), uM0_to_uM01_uM02),
    (('uM01',), ('uM0', 'uM02'), uM01_to_uM0_uM02),
    (('uM02',), ('uM0', 'uM01'), uM02_to_uM0_uM01),
    (('a', 'sini', 'n', 'e'), ('K',), a_sini_n_e_to_K), # FIXME Technically this may actually have a mass ratio dependence due to a being relative.
    (('a1', 'sini', 'n', 'e'), ('K1',), a_sini_n_e_to_K),
    (('a2', 'sini', 'n', 'e'), ('K2',), a_sini_n_e_to_K),
    (('K', 'n', 'e', 'a'), ('sini',), K_n_e_a_to_sini), # FIXME Technically this may actually have a mass ratio dependence due to a being relative.
    (('K1', 'n', 'e', 'a1'), ('sini',), K_n_e_a_to_sini),
    (('K2', 'n', 'e', 'a2'), ('sini',), K_n_e_a_to_sini),
    (('K', 'e', 'a', 'sini'), ('n',), K_e_a_sini_to_n), # FIXME Technically this may actually have a mass ratio dependence due to a being relative.
    (('K1', 'e', 'a1', 'sini'), ('n',), K_e_a_sini_to_n),
    (('K2', 'e', 'a2', 'sini'), ('n',), K_e_a_sini_to_n),
    (('K', 'a', 'sini', 'n'), ('e',), K_a_sini_n_to_e), # FIXME Technically this may actually have a mass ratio dependence due to a being relative.
    (('K1', 'a1', 'sini', 'n'), ('e',), K_a_sini_n_to_e),
    (('K2', 'a2', 'sini', 'n'), ('e',), K_a_sini_n_to_e),
    (('K', 'sini', 'n', 'e'), ('a',), K_sini_n_e_to_a), # FIXME Technically this may actually have a mass ratio dependence due to a being relative.
    (('K1', 'sini', 'n', 'e'), ('a1',), K_sini_n_e_to_a),
    (('K2', 'sini', 'n', 'e'), ('a2',), K_sini_n_e_to_a),
    (('M2', 'M1',), ('q',), div_ab),
    (('M1', 'q',), ('M2',), mul_ab),
    (('M2', 'q',), ('M1',), div_ab),
    (('Mtot', 'q',), ('M1', 'M2'), Mtot_q_to_M1_M2),
    (('n', 'K1', 'e'), ('a1sini',), n_K_e_to_asini),
    (('n', 'K2', 'e'), ('a2sini',), n_K_e_to_asini),
    (('a1sini', 'n', 'e'), ('K1',), asini_n_e_to_K),
    (('a2sini', 'n', 'e'), ('K2',), asini_n_e_to_K),
    (('K1', 'e', 'a1sini',), ('n',), K_e_asini_to_n),
    (('K2', 'e', 'a2sini',), ('n',), K_e_asini_to_n),
    (('n', 'a1sini', 'K1',), ('e',), n_asini_K_to_e),
    (('n', 'a2sini', 'K2',), ('e',), n_asini_K_to_e),
    (('M1', 'a1sini', 'M2sini',), ('a2',), Ma_aasini_Mbsini_to_ab),
    (('M2', 'a2sini', 'M1sini',), ('a1',), Ma_aasini_Mbsini_to_ab),
    (('a1sini', 'M2sini', 'a2',), ('M1',), aasini_Mbsini_ab_to_Ma),
    (('a2sini', 'M1sini', 'a1',), ('M2',), aasini_Mbsini_ab_to_Ma),
    (('M2sini', 'a2', 'M1'), ('a1sini',), Mbsini_ab_Ma_to_aasini),
    (('M1sini', 'a1', 'M2'), ('a2sini',), Mbsini_ab_Ma_to_aasini),
    (('a2', 'M1', 'a1sini',), ('M2sini',), ab_Ma_aasini_to_Mbsini),
    (('a1', 'M2', 'a2sini',), ('M1sini',), ab_Ma_aasini_to_Mbsini),
]

_TransformStep = tuple[tuple[str, ...], Callable, tuple[str, ...]]
_Plan = tuple[_TransformStep, ...]

_compiled_transform_graph = tuple(
    (inputs, outputs, fn, frozenset(inputs), frozenset(outputs))
    for inputs, outputs, fn in _transform_graph
)

_transforms_by_input: dict[str, list[int]] = defaultdict(list)
for transform_index, (_, _, _, input_set, _) in enumerate(_compiled_transform_graph):
    for name in input_set:
        _transforms_by_input[name].append(transform_index)

@lru_cache(maxsize=512)
def _covered_parameters_cached(known: frozenset[str]) -> frozenset[str]:
    """Return the graph closure without repeatedly scanning every transform."""
    covered = set(known)
    missing_input_count = [
        len(input_set.difference(covered))
        for _, _, _, input_set, _ in _compiled_transform_graph
    ]
    ready = deque(
        index for index, count in enumerate(missing_input_count) if count == 0
    )

    while ready:
        index = ready.popleft()
        _, _, _, _, output_set = _compiled_transform_graph[index]
        new_outputs = output_set.difference(covered)
        if not new_outputs:
            continue

        covered.update(new_outputs)
        for output in new_outputs:
            for dependent_index in _transforms_by_input.get(output, ()):
                missing_input_count[dependent_index] -= 1
                if missing_input_count[dependent_index] == 0:
                    ready.append(dependent_index)

    return frozenset(covered)


def _merge_plans(*plans: _Plan) -> _Plan:
    """Stable union of already-topological plans."""
    merged = []
    seen = set()
    for plan in plans:
        for step in plan:
            if step not in seen:
                seen.add(step)
                merged.append(step)
    return tuple(merged)


@lru_cache(maxsize=512)
def _plans_for_known(known: frozenset[str]) -> dict[str, _Plan]:
    """Find the lowest unique-transform-count plan for every reachable name."""
    plans: dict[str, _Plan] = {name: () for name in known}

    changed = True
    while changed:
        changed = False
        for inputs, outputs, fn, _, _ in _compiled_transform_graph:
            if not all(name in plans for name in inputs):
                continue

            step = (inputs, fn, outputs)
            candidate = _merge_plans(*(plans[name] for name in inputs), (step,))
            for output in outputs:
                # Explicit inputs always win. This also gives identity transforms
                # an empty execution plan.
                if output in known:
                    continue
                current = plans.get(output)
                if current is None or len(candidate) < len(current):
                    plans[output] = candidate
                    changed = True

    return plans


def covered_parameters(known_params: set):
    """
    From a list of known parameters, return a list of all parameters that can be known via mathematical relationships.
    """
    return set(_covered_parameters_cached(frozenset(known_params)))

def uncovered_parameters(known_params: set):
    """
    From a list of known parameters, return a list of all parameters that cannot be known without additional information.
    """
    covered = covered_parameters(known_params)
    return _all_parameters.difference(covered)

def shortest_path(known_params, end):
    """
    From a list of known parameters, return the shortest path to a desired parameter. If there is no such path, raises a KeyError.
    """
    return list(_plans_for_known(frozenset(known_params))[end])

@lru_cache(maxsize=512)
def _execution_plan(
    known: frozenset[str], targets: frozenset[str]
) -> _Plan:
    plans = _plans_for_known(known)
    for target in sorted(targets):
        if target not in plans:
            raise KeyError(target)

    merged = _merge_plans(*(plans[target] for target in sorted(targets)))

    # Merging independently optimal paths can include an alternate transform
    # whose outputs have already been produced. Remove such work while preserving
    # dependency order.
    available = set(known)
    pruned = []
    for step in merged:
        inputs, _, outputs = step
        if not all(name in available for name in inputs):
            raise RuntimeError("Transform plan is not topologically ordered")
        if any(name not in available for name in outputs):
            pruned.append(step)
            available.update(outputs)
    return tuple(pruned)


def _run_plan(plan: _Plan, kwargs: dict) -> dict:
    params = kwargs.copy()
    for inputs, fn, outputs in plan:
        result = fn(*(params[name] for name in inputs))
        values = (result,) if len(outputs) == 1 else result
        for name, value in zip(outputs, values):
            # Do not replace an explicit input or an earlier, cheaper derivation.
            params.setdefault(name, value)
    return params


@lru_cache(maxsize=512)
def _build_transform_function_cached(known: frozenset[str], end: str):
    plan = _execution_plan(known, frozenset((end,)))

    if not plan:
        def transform_function(**kwargs):
            return kwargs[end]
    else:
        def transform_function(**kwargs):
            return _run_plan(plan, kwargs)[end]

    transform_function.__doc__ = f"Transforms {sorted(known)} -> {end}"
    return transform_function

def build_transform_function(known_params, end):
    """
    From a list of known parameters, return a function that transforms the known parameters into the desired parameter. If there is no such path, raises a KeyError.
    """
    return _build_transform_function_cached(frozenset(known_params), end)

@lru_cache(maxsize=512)
def _build_transform_functions_cached(
    known: frozenset[str], targets: tuple[str, ...]
):
    plan = _execution_plan(known, frozenset(targets))

    if not plan:
        def transform_functions(**kwargs):
            return {target: kwargs[target] for target in targets}
    else:
        def transform_functions(**kwargs):
            params = _run_plan(plan, kwargs)
            return {target: params[target] for target in targets}

    transform_functions.__doc__ = (
        f"Transforms {sorted(known)} -> {list(targets)}"
    )
    return transform_functions


def build_transform_functions(known_params, ends):
    """Build one callable that derives all requested parameters once."""
    # Preserve caller order in the result while removing duplicate targets.
    targets = tuple(dict.fromkeys(ends))
    return _build_transform_functions_cached(frozenset(known_params), targets)

def overconstrained_parameters(known_params) -> set[str]:
    known_params = frozenset(known_params)
    overconstrained = set()
    for param in known_params:
        covered = covered_parameters(known_params - {param})
        if param in covered:
            overconstrained.add(param)
    return overconstrained

