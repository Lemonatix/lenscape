"""H0 from time-delay lenses: grid posterior in flat ΛCDM or wCDM.

In a flat universe all distances scale exactly with c/H0:

    D_d  = c/H0 * chi_d / (1 + z_d)
    D_dt = (1 + z_d) D_d D_s / D_ds = c/H0 * chi_d chi_s / (chi_s - chi_d)

with the dimensionless comoving distances chi(z; Omega_m, w). One integration
per (Omega_m, w) is therefore enough; the H0 axis is a pure rescaling. Priors
as in Wong et al. (2020): H0 uniform, Omega_m in [0.05, 0.5], w in [-2.5, 0.5].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cosmography import C_KMS, _comoving_table
from .h0licow import TimeDelayLens


@dataclass
class H0Posterior:
    H0: np.ndarray
    Om: np.ndarray
    w: np.ndarray            # length 1 for ΛCDM
    logpost: np.ndarray      # shape (len(w), len(Om), len(H0))

    def marginal_H0(self) -> np.ndarray:
        p = np.exp(self.logpost - np.nanmax(self.logpost))
        m = np.nansum(p, axis=(0, 1))
        return m / np.trapezoid(m, self.H0)

    def summary(self, q=(0.16, 0.5, 0.84)) -> tuple[float, float, float]:
        """(median, -error, +error) of the marginal H0 distribution."""
        m = self.marginal_H0()
        cdf = np.concatenate([[0], np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(self.H0))])
        cdf /= cdf[-1]
        lo, med, hi = np.interp(q, cdf, self.H0)
        return float(med), float(med - lo), float(hi - med)


def _distances(lens: TimeDelayLens, Om: float, w: float):
    z, chi = _comoving_table(Om, w, lens.z_s + 0.01, n=4096)
    chi_d = np.interp(lens.z_d, z, chi)
    chi_s = np.interp(lens.z_s, z, chi)
    dd_unit = chi_d / (1 + lens.z_d)                  # D_d in units of c/H0
    ddt_unit = chi_d * chi_s / (chi_s - chi_d)        # D_dt in units of c/H0
    return dd_unit, ddt_unit


def infer_h0(lenses: list[TimeDelayLens],
             H0=np.arange(50.0, 100.0001, 0.1),
             Om=np.linspace(0.05, 0.5, 46),
             w=np.array([-1.0])) -> H0Posterior:
    """Joint grid posterior of all given lenses."""
    H0 = np.asarray(H0, float)
    Om = np.asarray(Om, float)
    w = np.asarray(w, float)
    logpost = np.zeros((len(w), len(Om), len(H0)))
    scale = C_KMS / H0  # Mpc
    for lens in lenses:
        for j, wj in enumerate(w):
            for i, om in enumerate(Om):
                dd_u, ddt_u = _distances(lens, om, wj)
                logpost[j, i] += lens.loglike(dd_u * scale, ddt_u * scale)
    return H0Posterior(H0, Om, w, logpost)
