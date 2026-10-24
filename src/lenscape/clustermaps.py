"""Gitterbasierte Massenmodelle, z. B. die Hubble-Frontier-Fields-Karten.

Eine Karte liefert die Konvergenz kappa_inf(theta) (und meist die Ablenkwinkel)
für eine Quelle im Unendlichen, also D_ds/D_s = 1. Für eine Quelle bei z_s
werden kappa und alpha mit dem Faktor D_ds/D_s(z_d, z_s) skaliert.

Fehlen die Ablenkkarten, folgen sie aus kappa durch Faltung

    alpha(theta) = 1/pi * ∫ kappa(theta') (theta - theta') / |theta - theta'|^2 d^2theta',

hier mit FFT auf einem doppelt so großen, mit Nullen aufgefüllten Gitter.

Koordinaten: x, y in Bogensekunden entlang der Pixelachsen der Karte, mit dem
Kartenzentrum als Ursprung (bei HFF-Karten ist Norden oben, Osten links).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator

from .cosmology import LensGeometry
from .lenses import Lens


def deflection_from_kappa(kappa: np.ndarray, pixscale: float) -> tuple[np.ndarray, np.ndarray]:
    """alpha_x, alpha_y (arcsec) aus einer kappa-Karte per FFT-Faltung."""
    ny, nx = kappa.shape
    py, px = 2 * ny, 2 * nx
    yy = (np.arange(py) - ny) * pixscale
    xx = (np.arange(px) - nx) * pixscale
    X, Y = np.meshgrid(xx, yy)
    r2 = X**2 + Y**2
    with np.errstate(divide="ignore", invalid="ignore"):
        kx = np.where(r2 > 0, X / (np.pi * r2), 0.0)
        ky = np.where(r2 > 0, Y / (np.pi * r2), 0.0)
    # Kern so verschieben, dass der Nullpunkt bei Index (0, 0) liegt
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    pad = np.zeros((py, px))
    pad[:ny, :nx] = kappa
    fk = np.fft.rfft2(pad)
    area = pixscale**2
    ax = np.fft.irfft2(fk * np.fft.rfft2(kx), s=(py, px))[:ny, :nx] * area
    ay = np.fft.irfft2(fk * np.fft.rfft2(ky), s=(py, px))[:ny, :nx] * area
    return ax, ay


@dataclass
class MassMap(Lens):
    """Linse aus Karten auf einem regelmäßigen Gitter (Normierung D_ds/D_s = 1)."""

    kappa_inf: np.ndarray
    pixscale: float                      # arcsec pro Pixel
    alpha_x_inf: np.ndarray | None = None
    alpha_y_inf: np.ndarray | None = None
    ratio: float = 1.0                   # D_ds/D_s der gewünschten Quellrotverschiebung
    name: str = ""

    def __post_init__(self):
        self.kappa_inf = np.nan_to_num(np.asarray(self.kappa_inf, dtype=float))
        if self.alpha_x_inf is None or self.alpha_y_inf is None:
            self.alpha_x_inf, self.alpha_y_inf = deflection_from_kappa(self.kappa_inf, self.pixscale)
        ny, nx = self.kappa_inf.shape
        self.x = (np.arange(nx) - (nx - 1) / 2) * self.pixscale
        self.y = (np.arange(ny) - (ny - 1) / 2) * self.pixscale
        kw = dict(bounds_error=False, fill_value=np.nan)
        self._k = RegularGridInterpolator((self.y, self.x), self.kappa_inf, **kw)
        self._ax = RegularGridInterpolator((self.y, self.x), np.nan_to_num(self.alpha_x_inf), **kw)
        self._ay = RegularGridInterpolator((self.y, self.x), np.nan_to_num(self.alpha_y_inf), **kw)

    # ------------------------------------------------------------- Konstruktion
    @classmethod
    def from_fits(cls, kappa_file: Path | str, alpha_x_file=None, alpha_y_file=None, name: str = "") -> "MassMap":
        from astropy.io import fits
        from astropy.wcs import WCS

        with fits.open(kappa_file) as h:
            hdu = next(x for x in h if x.data is not None)
            kappa = np.asarray(hdu.data, dtype=float)
            wcs = WCS(hdu.header).celestial
            pixscale = float(abs(wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value))
        ax = ay = None
        if alpha_x_file and alpha_y_file:
            ax = np.asarray(fits.getdata(alpha_x_file), dtype=float)
            ay = np.asarray(fits.getdata(alpha_y_file), dtype=float)
        return cls(kappa, pixscale, ax, ay, name=name)

    def at_source_redshift(self, geom: LensGeometry) -> "MassMap":
        """Gleiche Karte, skaliert auf die Quelle von ``geom`` (D_ds/D_s)."""
        ratio = float((geom.D_ds / geom.D_s).decompose().value)
        m = MassMap.__new__(MassMap)
        m.__dict__.update(self.__dict__)
        m.ratio = ratio
        return m

    # ---------------------------------------------------------------- Lens-API
    def _pts(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        return np.column_stack([y.ravel(), x.ravel()]), x.shape

    def deflection(self, x, y):
        pts, shape = self._pts(x, y)
        return (self.ratio * self._ax(pts).reshape(shape), self.ratio * self._ay(pts).reshape(shape))

    def convergence(self, x, y):
        pts, shape = self._pts(x, y)
        return self.ratio * self._k(pts).reshape(shape)

    # ---------------------------------------------------------- Kartengrößen
    @property
    def extent(self) -> list[float]:
        h = self.pixscale / 2
        return [self.x[0] - h, self.x[-1] + h, self.y[0] - h, self.y[-1] + h]

    def grid(self, step: int = 1):
        X, Y = np.meshgrid(self.x[::step], self.y[::step])
        return X, Y

    def magnification_grid(self, step: int = 1):
        """|det A|^-1 auf dem (ausgedünnten) Kartengitter, aus Ableitungen der Ablenkkarten."""
        ax = self.ratio * self.alpha_x_inf[::step, ::step]
        ay = self.ratio * self.alpha_y_inf[::step, ::step]
        d = self.pixscale * step
        dax_dy, dax_dx = np.gradient(ax, d)
        day_dy, day_dx = np.gradient(ay, d)
        det = (1 - dax_dx) * (1 - day_dy) - dax_dy * day_dx
        with np.errstate(divide="ignore"):
            return 1.0 / det


def sigma_crit_inf(z_d: float, cosmo) -> u.Quantity:
    """Kritische Flächendichte für D_ds/D_s = 1: c^2 / (4 pi G D_d)."""
    D_d = cosmo.angular_diameter_distance(z_d)
    return (const.c**2 / (4 * np.pi * const.G * D_d)).to(u.M_sun / u.kpc**2)


def enclosed_mass_profile(m: MassMap, z_d: float, cosmo, radii_kpc, center=(0.0, 0.0)) -> np.ndarray:
    """Projizierte Masse M(<R) in M_sun um ``center`` (arcsec) für die Radien in kpc."""
    kpc_per_arcsec = (cosmo.angular_diameter_distance(z_d) * (np.pi / 180 / 3600)).to(u.kpc).value
    X, Y = m.grid()
    R = np.hypot(X - center[0], Y - center[1]) * kpc_per_arcsec
    sig = sigma_crit_inf(z_d, cosmo).value  # M_sun/kpc^2
    pix_area = (m.pixscale * kpc_per_arcsec) ** 2
    mass_pix = m.kappa_inf * sig * pix_area
    order = np.argsort(R.ravel())
    cum = np.cumsum(mass_pix.ravel()[order])
    return np.interp(np.asarray(radii_kpc, float), R.ravel()[order], cum)


def effective_einstein_radius(m: MassMap, step: int = 1) -> float:
    """θ_E,eff = sqrt(A/π) mit A = Fläche innerhalb der tangentialen kritischen Kurve.

    Gesucht wird die größte geschlossene Kontur det A = 0; das ist bei Haufen die
    äußere (tangentiale) kritische Kurve. Gängiges Maß für die Linsenstärke eines Haufens.
    """
    import contourpy

    X, Y = m.grid(step)
    mu = m.magnification_grid(step)
    det = 1.0 / mu
    gen = contourpy.contour_generator(X, Y, det)
    best = 0.0
    for line in gen.lines(0.0):
        if len(line) < 10 or not np.allclose(line[0], line[-1], atol=2 * m.pixscale * step):
            continue  # nur geschlossene Kurven
        x, y = line[:, 0], line[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        best = max(best, area)
    return float(np.sqrt(best / np.pi))
