"""Likelihoods for measured distances (D_dt and optionally D_d) of individual lenses.

Lens modelling yields posterior distributions of the distances. To evaluate a
cosmology one needs their density at the point the cosmology predicts. As in
H0LiCOW (Wong et al. 2020, code by Bonvin & Millon) there are three forms:

* skewed log-normal distribution (analytic fit, e.g. B1608+656)
* 1D kernel density estimate on D_dt samples (GLEE analyses)
* 2D kernel density estimate on (D_d, D_dt) samples (lenstronomy analyses)

The kernel densities are computed on a binned grid (histogram, then Gaussian
smoothing) and interpolated -- the same idea as the "kde_hist" variant of
H0LiCOW, but vectorized and therefore fast enough for grid scans.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d

_NEG = -np.inf


@dataclass
class SkewLogNormal:
    """p(x) = LogNormal(x - lam; mu, sigma) for x > lam."""

    mu: float
    sigma: float
    lam: float

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        y = x - self.lam
        out = np.full(x.shape, _NEG)
        ok = y > 0
        ly = np.log(y[ok])
        out[ok] = -((ly - self.mu) ** 2) / (2 * self.sigma**2) - np.log(np.sqrt(2 * np.pi) * self.sigma * y[ok])
        return out


class KDE1D:
    """Binned Gaussian kernel density of a (weighted) sample."""

    def __init__(self, samples, weights=None, bandwidth: float = 20.0, nbins: int = 400, pad: float = 6.0):
        samples = np.asarray(samples, dtype=float)
        lo = samples.min() - pad * bandwidth
        hi = samples.max() + pad * bandwidth
        # fine grid: at least 4 points per bandwidth
        n = max(nbins, int((hi - lo) / bandwidth * 4))
        hist, edges = np.histogram(samples, bins=n, range=(lo, hi), weights=weights, density=True)
        dx = edges[1] - edges[0]
        dens = gaussian_filter1d(hist, bandwidth / dx, mode="constant")
        self.x = 0.5 * (edges[1:] + edges[:-1])
        with np.errstate(divide="ignore"):
            self.logp = np.log(np.clip(dens, 0, None))

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.interp(x, self.x, self.logp, left=_NEG, right=_NEG)
        return np.where(np.isfinite(out), out, _NEG)


class KDE2D:
    """Binned Gaussian kernel density in (D_d, D_dt)."""

    def __init__(self, dd, ddt, weights=None, bandwidth=(20.0, 20.0), nbins: int = 80, pad: float = 6.0):
        dd = np.asarray(dd, dtype=float)
        ddt = np.asarray(ddt, dtype=float)
        bw = np.broadcast_to(np.asarray(bandwidth, dtype=float), (2,))
        rng = [(dd.min() - pad * bw[0], dd.max() + pad * bw[0]),
               (ddt.min() - pad * bw[1], ddt.max() + pad * bw[1])]
        # grid fine enough for the bandwidth, but at least nbins per axis
        n = [max(nbins, int((r[1] - r[0]) / b * 3)) for r, b in zip(rng, bw)]
        n = [min(k, 1500) for k in n]
        hist, ed, et = np.histogram2d(dd, ddt, bins=n, range=rng, weights=weights, density=True)
        sig = (bw[0] / (ed[1] - ed[0]), bw[1] / (et[1] - et[0]))
        dens = gaussian_filter(hist, sig, mode="constant")
        with np.errstate(divide="ignore"):
            logp = np.log(np.clip(dens, 0, None))
        logp[~np.isfinite(logp)] = -1e300
        self._interp = RegularGridInterpolator(
            (0.5 * (ed[1:] + ed[:-1]), 0.5 * (et[1:] + et[:-1])), logp,
            bounds_error=False, fill_value=-1e300,
        )

    def logpdf(self, dd, ddt):
        dd, ddt = np.broadcast_arrays(np.asarray(dd, float), np.asarray(ddt, float))
        pts = np.column_stack([dd.ravel(), ddt.ravel()])
        out = self._interp(pts).reshape(dd.shape)
        return np.where(out < -1e299, _NEG, out)


def scott_bandwidth(samples, n_dim: int) -> float:
    """Scott's rule as in scipy.stats.gaussian_kde: sigma * n^(-1/(d+4))."""
    samples = np.asarray(samples, dtype=float)
    return float(np.std(samples, ddof=1) * len(samples) ** (-1.0 / (n_dim + 4)))
