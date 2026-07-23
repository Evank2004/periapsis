import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration dependency
    njit = None


def kep_guess(e,M):
    t34 = e**2
    t35 = e * t34
    t33 = np.cos(M)
    return M + (-0.5 *t35 + e + (t34 + 1.5*t33*t35)*t33)*np.sin(M)

# this is now a helper function to solve Keplers equation
#instead of using the newton-raphson method, we use 
# Halleys method here with the second derivative and third derivative 
# to account for curvature in the function
def e_help(e,M,x):
    t1=np.cos(x)
    t2=np.sin(x)
    t3 = -1 + e*t1 # negative first derivative to account for subtraction in function
    t4 = e*t2
    t5 = -x + t4 + M
    t6 = t5/(0.5*t5*t4/t3 + t3)
    return t5/((0.5*t2 - (1/6)*t1*t6)*e*t6+t3)


def _solve_kepler_numpy(M, e):
    M_array = np.asarray(M, dtype=np.float64)
    e_array = np.asarray(e, dtype=np.float64)
    flat_M = np.mod(M_array.ravel(), 2.0 * np.pi)
    flat_e = e_array.ravel()
    flat_E = np.empty_like(flat_M)

    for i in range(flat_M.size):
        Mi = flat_M[i]
        ei = flat_e[i] if len(flat_e) > 1 else flat_e[0]
        E = kep_guess(ei, Mi)
        for _ in range(3):
            E = E - e_help(ei, Mi, E)
        flat_E[i] = E

    return flat_E.reshape(M_array.shape)


if njit is not None:

    @njit(cache=True)
    def _kep_guess_numba(e, M):
        t34 = e ** 2
        t35 = e * t34
        t33 = np.cos(M)
        return M + (-0.5 * t35 + e + (t34 + 1.5 * t33 * t35) * t33) * np.sin(M)


    @njit(cache=True)
    def _e_help_numba(e, M, x):
        t1 = np.cos(x)
        t2 = np.sin(x)
        t3 = -1.0 + e * t1
        t4 = e * t2
        t5 = -x + t4 + M
        t6 = t5 / (0.5 * t5 * t4 / t3 + t3)
        return t5 / ((0.5 * t2 - (1.0 / 6.0) * t1 * t6) * e * t6 + t3)


    @njit(cache=True)
    def _solve_kepler_numba(flat_M, flat_e):
        flat_E = np.empty_like(flat_M)
        two_pi = 2.0 * np.pi

        for i in range(flat_M.size):
            Mi = flat_M[i] % two_pi
            ei = flat_e[i] if len(flat_e) > 1 else flat_e[0]
            E = _kep_guess_numba(ei, Mi)
            for _ in range(3):
                E = E - _e_help_numba(ei, Mi, E)
            flat_E[i] = E

        return flat_E

def solve_kepler(M, e):
    """
    Solves Kepler's equation M = E - e*sin(E) for the eccentric anomaly E given mean anomaly M and eccentricity e.
    
    Parameters:
    M : float or array-like
        Mean anomaly in radians.
    e : float
        Eccentricity of the orbit (0 <= e < 1).
    
    Returns:
    E : float or array-like
        Eccentric anomaly in radians, matching the shape of M.
    """
    M_array = np.asarray(M, dtype=np.float64)
    flat_M = M_array.ravel()

    e_array = np.asarray(e, dtype=np.float64)
    flat_e = e_array.ravel()

    if njit is None:
        return _solve_kepler_numpy(M_array, e_array)

    flat_E = _solve_kepler_numba(flat_M, flat_e)
    return flat_E.reshape(M_array.shape)


def transform_theile(A,B,F,G):
    'Transforms Thiele-Innes parameters into Campbell parameters a1,i,w,long'
    #need to transform A B F G into our parameters a, i, long,w
    
    popovic_k = (A*A + B*B + F*F + G*G) / 2.0
    popovic_m = A*G - B*F
    popovic_j = np.sqrt(np.maximum(0, popovic_k * popovic_k - popovic_m*popovic_m))
    a1 = np.sqrt(popovic_j + popovic_k) # as
    i = np.arctan2(a1 * np.sqrt(2.0 * popovic_j), popovic_m)
    vpi = np.arctan2(B-F, A+G)
    Omo = np.arctan2(B+F, A-G)
    long = np.mod((vpi + Omo) / 2.0 + 2*np.pi, np.pi)
    w = np.mod(vpi - long + 2*np.pi, 2*np.pi)
    

    return a1,i,w,long

def campbell_to_thiele(a1,cosi,w,long):
    # Convert Campbell parameters to Thiele-Innes parameters
    cO, sO = np.cos(long), np.sin(long)
    cw, sw = np.cos(w), np.sin(w)

    

    A = a1 * (cO * cw - sO * sw * cosi)
    B = a1 * (sO * cw + cO * sw * cosi)
    F = a1 * (-cO * sw - sO * cw * cosi)
    G = a1 * (-sO * sw + cO * cw * cosi)

    return A, B, F, G

def solve_mass(a1,P,m1,plx=None,max_iter=15,tol=1e-6):
    'Solves for the secondary mass M2 given the'
    'primary mass M1, and the samples for semimajor axis a1 and period P'
    if plx is not None:
        a1 = a1 / plx #convert to AU
    Ps23 = P**(2./3.)

    #initial guess 
    K = a1 / (Ps23 * m1**(1./3.))
    K = np.clip(K,0.0,None)
    M = m1 * (1. + 1.4*K**1.135 + 0.743*K**3.163)
    f_M = a1**3. / P**2. #mass function

    for i in range(max_iter):
        Ps23divM23 = Ps23 * M**(-2./3.)

        f  = (M - m1) * Ps23divM23 - a1
        df = Ps23divM23 * ((1./3.) + (2./3.)*m1/M)
        delta = f / df
        M -= delta
        if np.all(np.abs(delta) < tol):
            break
    else:
        not_converged = (np.abs(delta) > tol) & np.isfinite(delta)
        if np.any(not_converged):
            max_delta = np.nanmax(np.abs(delta[not_converged]))
            print(
                f"Warning: max iterations reached without full convergence "
                f"on {np.sum(not_converged)} inputs. "
                f"Max |Δ| = {max_delta}"
            )
    M2 = M - m1
    return M2,f_M


def gaia_single_motion(spsi,cpsi,t,plx_fac,x,err):
    """
    Computes single star motion for Gaia data"""
    
    A = np.column_stack([spsi,cpsi,plx_fac,spsi*t,cpsi*t])

    w = 1.0 / err
    x_w = x * w
    A_w = A * w[:, None]

    ATA = A_w.T @ A_w
    ATx = A_w.T @ x_w

    mu = np.linalg.solve(ATA, ATx)

    cov_mu = np.linalg.inv(ATA)

    mu_err = np.sqrt(np.diag(cov_mu))


    model_werr = A_w @ mu

    residuals = x_w - model_werr
    chi2 = np.sum(residuals**2)

    dof = len(x) - len(mu)
    
    return {
        "mu": mu,
        "mu_err": mu_err,
        "chi2": chi2,
        "dof": dof
    }