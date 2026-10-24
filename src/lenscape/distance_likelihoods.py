"""Likelihoods für gemessene Distanzen (D_dt und optional D_d) einzelner Linsen.

Die Linsenmodellierung liefert Posterior-Verteilungen der Distanzen. Um eine
Kosmologie zu bewerten, braucht man ihre Dichte an der von der Kosmologie
vorhergesagten Stelle. Wie bei H0LiCOW (Wong et al. 2020, Code von Bonvin &
Millon) gibt es drei Formen:

* schiefe Log-Normalverteilung (analytischer Fit, z. B. B1608+656)
* 1D-Kerndichteschätzer auf D_dt-Samples (GLEE-Analysen)
* 2D-Kerndichteschätzer auf (D_d, D_dt)-Samples (lenstronomy-Analysen)

Die Kerndichten werden auf einem gebinnten Gitter berechnet (Histogramm, dann
Gauß-Glättung) und interpoliert -- das ist dieselbe Idee wie die "kde_hist"-
Variante von H0LiCOW, aber vektorisiert und damit schnell genug für Gitterscans.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d

_NEG = -np.inf


@dataclass
class SkewLogNormal:
    """p(x) = LogNormal(x - lam; mu, sigma) für x > lam."""

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
    """Gebinnte Gauß-Kerndichte einer (gewichteten) Stichprobe."""

    def __init__(self, samples, weights=None, bandwidth: float = 20.0, nbins: int = 400, pad: float = 6.0):
        samples = np.asarray(samples, dtype=float)
        lo = samples.min() - pad * bandwidth
        hi = samples.max() + pad * bandwidth
        # feines Gitter: mindestens 4 Punkte pro Bandbreite
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
    """Gebinnte Gauß-Kerndichte in (D_d, D_dt)."""

    def __init__(self, dd, ddt, weights=None, bandwidth=(20.0, 20.0), nbins: int = 80, pad: float = 6.0):
        dd = np.asarray(dd, dtype=float)
        ddt = np.asarray(ddt, dtype=float)
        bw = np.broadcast_to(np.asarray(bandwidth, dtype=float), (2,))
        rng = [(dd.min() - pad * bw[0], dd.max() + pad * bw[0]),
               (ddt.min() - pad * bw[1], ddt.max() + pad * bw[1])]
        # Gitter fein genug für die Bandbreite, aber mindestens nbins pro Achse
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
    """Scotts Regel wie in scipy.stats.gaussian_kde: sigma * n^(-1/(d+4))."""
    samples = np.asarray(samples, dtype=float)
    return float(np.std(samples, ddof=1) * len(samples) ** (-1.0 / (n_dim + 4)))
