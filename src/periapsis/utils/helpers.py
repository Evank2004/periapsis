from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from astropy.time import Time
import numpy as np

def _helper_for_periodogram(A,x,err):
    w = 1.0 / err
    x_w = x * w
    A_w = A * w[:, None]

    ATA = A_w.T @ A_w
    ATx = A_w.T @ x_w

    try:
        mu = np.linalg.solve(ATA, ATx)
    except np.linalg.LinAlgError:
        mu, _, _, _ = np.linalg.lstsq(A_w, x_w, rcond=None) # [delta alpha,delta delta, parallax,mu_alpha,mu_delta]

    model_werr = A_w @ mu

    residuals = x_w - model_werr
    chi2 = np.sum(residuals**2)

    return mu, chi2



