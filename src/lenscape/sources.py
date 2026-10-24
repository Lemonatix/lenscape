"""Flächenhelligkeitsprofile für Hintergrundquellen (Winkel in arcsec)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaincinv


def _elliptical_radius(x, y, x0, y0, q, phi):
    dx = np.asarray(x, dtype=float) - x0
    dy = np.asarray(y, dtype=float) - y0
    c, s = np.cos(phi), np.sin(phi)
    xr = c * dx + s * dy
    yr = -s * dx + c * dy
    return np.sqrt(q * xr**2 + yr**2 / q)


class Source:
    def brightness(self, x, y):
        raise NotImplementedError


@dataclass
class GaussianSource(Source):
    x0: float = 0.0
    y0: float = 0.0
    sigma: float = 0.1
    amp: float = 1.0
    q: float = 1.0
    phi: float = 0.0

    def brightness(self, x, y):
        r = _elliptical_radius(x, y, self.x0, self.y0, self.q, self.phi)
        return self.amp * np.exp(-0.5 * (r / self.sigma) ** 2)


@dataclass
class SersicSource(Source):
    """Sérsic-Profil I(r) = I_e exp(-b_n [(r/R_e)^(1/n) - 1]); n=1 Scheibe, n=4 de Vaucouleurs."""

    x0: float = 0.0
    y0: float = 0.0
    R_e: float = 0.2
    n: float = 1.0
    amp: float = 1.0
    q: float = 1.0
    phi: float = 0.0

    def brightness(self, x, y):
        r = _elliptical_radius(x, y, self.x0, self.y0, self.q, self.phi)
        b_n = gammaincinv(2 * self.n, 0.5)
        return self.amp * np.exp(-b_n * ((r / self.R_e) ** (1 / self.n) - 1))


@dataclass
class ImageSource(Source):
    """Ein beliebiges Bild als Quelle; ``extent`` = Kantenlänge in arcsec, zentriert auf (x0, y0)."""

    image: np.ndarray
    extent: float = 2.0
    x0: float = 0.0
    y0: float = 0.0

    def brightness(self, x, y):
        from scipy.ndimage import map_coordinates

        img = np.asarray(self.image, dtype=float)
        ny, nx = img.shape[:2]
        u = (np.asarray(x) - self.x0) / self.extent + 0.5
        v = 0.5 - (np.asarray(y) - self.y0) / self.extent
        cols = u * (nx - 1)
        rows = v * (ny - 1)
        coords = [rows.ravel(), cols.ravel()]
        if img.ndim == 2:
            out = map_coordinates(img, coords, order=1, cval=0.0)
            return out.reshape(np.shape(x))
        chans = [map_coordinates(img[..., k], coords, order=1, cval=0.0) for k in range(img.shape[2])]
        return np.stack(chans, axis=-1).reshape(*np.shape(x), img.shape[2])
