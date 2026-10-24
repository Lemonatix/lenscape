"""Kosmographie mit Einsteinradien: Dunkle Energie aus einer Linsenstichprobe.

Für eine isotherme Linse gilt

    theta_E = 4 pi (sigma_SIE / c)^2 * D_ds / D_s .

Misst man theta_E (aus dem Bild) und sigma (aus dem Spektrum), folgt pro Linse
ein *beobachtetes* Distanzverhältnis

    R_obs = theta_E c^2 / (4 pi sigma^2) = f_E^2 * D_ds/D_s (z_d, z_s; Omega_m, w).

f_E = sigma_SIE / sigma ~ 1 fasst zusammen, wie gut die Stern-Geschwindigkeits-
dispersion die Gesamtmasse beschreibt; es wird als freier Parameter mitgefittet.
Im flachen Universum ist D_ds/D_s = 1 - chi_d/chi_s, unabhängig von H0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_KMS = 299792.458
ARCSEC = np.pi / 180 / 3600


def observed_distance_ratio(theta_E_arcsec, sigma_kms):
    """R_obs = theta_E c^2 / (4 pi sigma^2)."""
    return np.asarray(theta_E_arcsec) * ARCSEC * C_KMS**2 / (4 * np.pi * np.asarray(sigma_kms) ** 2)


def _comoving_table(Om, w, z_max, n=2048):
    """chi(z) (in Einheiten c/H0) für flaches wCDM auf einem z-Gitter."""
    z = np.linspace(0.0, z_max, n)
    E = np.sqrt(Om * (1 + z) ** 3 + (1 - Om) * (1 + z) ** (3 * (1 + w)))
    inv = 1.0 / E
    chi = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(z))])
    return z, chi


def model_distance_ratio(z_d, z_s, Om, w):
    """D_ds / D_s im flachen wCDM."""
    z_d = np.asarray(z_d, float)
    z_s = np.asarray(z_s, float)
    z, chi = _comoving_table(Om, w, float(np.max(z_s)) + 0.01)
    chi_d = np.interp(z_d, z, chi)
    chi_s = np.interp(z_s, z, chi)
    return 1.0 - chi_d / chi_s


@dataclass
class GridResult:
    Om: np.ndarray        # Gitterachse Omega_m
    w: np.ndarray         # Gitterachse w
    lnL: np.ndarray       # Profil-Log-Likelihood, Form (len(w), len(Om))
    lnf2: np.ndarray      # best-fit ln f_E^2 je Gitterpunkt
    sigma_int: np.ndarray  # best-fit intrinsische Streuung je Gitterpunkt

    @property
    def best(self) -> dict:
        j, i = np.unravel_index(np.argmax(self.lnL), self.lnL.shape)
        return {
            "Om": float(self.Om[i]),
            "w": float(self.w[j]),
            "f_E": float(np.exp(0.5 * self.lnf2[j, i])),
            "sigma_int": float(self.sigma_int[j, i]),
        }

    def delta_chi2(self) -> np.ndarray:
        return -2 * (self.lnL - self.lnL.max())


def fit_flat_wcdm(theta_E, sigma, e_sigma, z_d, z_s, e_theta_frac: float = 0.05,
                  Om=np.linspace(0.02, 1.0, 50), w=np.linspace(-3.0, 0.0, 61),
                  sigma_int=np.linspace(0.0, 0.4, 41)) -> GridResult:
    """Gitter-Likelihood in (Omega_m, w) für flaches wCDM.

    Gearbeitet wird in ln R: ln R_obs = ln(D_ds/D_s) + ln f_E^2 + Rauschen mit
    Varianz s_i^2 = (2 e_sigma/sigma)^2 + e_theta_frac^2 + sigma_int^2.
    ln f_E^2 wird analytisch, sigma_int numerisch herausprofiliert.
    """
    y = np.log(observed_distance_ratio(theta_E, sigma))
    s_meas2 = (2 * np.asarray(e_sigma) / np.asarray(sigma)) ** 2 + e_theta_frac**2

    lnL = np.full((len(w), len(Om)), -np.inf)
    lnf2 = np.zeros_like(lnL)
    sint = np.zeros_like(lnL)
    for j, wj in enumerate(w):
        for i, om in enumerate(Om):
            m = np.log(model_distance_ratio(z_d, z_s, om, wj))
            r = y - m
            best = (-np.inf, 0.0, 0.0)
            for si in sigma_int:
                v = s_meas2 + si**2
                wt = 1 / v
                off = np.sum(wt * r) / np.sum(wt)
                ll = -0.5 * np.sum(wt * (r - off) ** 2 + np.log(2 * np.pi * v))
                if ll > best[0]:
                    best = (ll, off, si)
            lnL[j, i], lnf2[j, i], sint[j, i] = best
    return GridResult(np.asarray(Om), np.asarray(w), lnL, lnf2, sint)
