"""Massenmodelle für dünne Linsen.

Alle Winkel sind in Bogensekunden. Jedes Modell liefert den *skalierten*
Ablenkwinkel alpha(theta) = D_ds/D_s * alpha_hat, sodass die Linsengleichung
einfach

    beta = theta - alpha(theta)

lautet. Die Konvergenz kappa = Sigma / Sigma_crit erfüllt div(alpha) = 2 kappa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cosmology import LensGeometry

_EPS = 1e-12


def _rotate(x, y, phi):
    """Drehe Koordinaten um -phi (ins Hauptachsensystem)."""
    c, s = np.cos(phi), np.sin(phi)
    return c * x + s * y, -s * x + c * y


def _rotate_back(ax, ay, phi):
    c, s = np.cos(phi), np.sin(phi)
    return c * ax - s * ay, s * ax + c * ay


class Lens:
    """Basisklasse. Unterklassen implementieren ``deflection`` und ``convergence``."""

    def deflection(self, x, y):
        raise NotImplementedError

    def convergence(self, x, y):
        raise NotImplementedError

    # --------------------------------------------------------- abgeleitete Größen
    def hessian(self, x, y, h: float = 1e-4):
        """Jacobi-Matrix von alpha per zentralem Differenzenquotienten.

        Rückgabe: (a_xx, a_xy, a_yx, a_yy)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        axp, ayp = self.deflection(x + h, y)
        axm, aym = self.deflection(x - h, y)
        bxp, byp = self.deflection(x, y + h)
        bxm, bym = self.deflection(x, y - h)
        a_xx = (axp - axm) / (2 * h)
        a_yx = (ayp - aym) / (2 * h)
        a_xy = (bxp - bxm) / (2 * h)
        a_yy = (byp - bym) / (2 * h)
        return a_xx, a_xy, a_yx, a_yy

    def jacobian_det(self, x, y):
        """det A mit A = d beta / d theta = 1 - d alpha / d theta."""
        a_xx, a_xy, a_yx, a_yy = self.hessian(x, y)
        return (1 - a_xx) * (1 - a_yy) - a_xy * a_yx

    def magnification(self, x, y):
        det = self.jacobian_det(x, y)
        return 1.0 / np.where(np.abs(det) < _EPS, np.sign(det) * _EPS + _EPS, det)

    def ray_shoot(self, x, y):
        """Bildebene -> Quellebene (Linsengleichung)."""
        ax, ay = self.deflection(x, y)
        return np.asarray(x) - ax, np.asarray(y) - ay

    def __add__(self, other: "Lens") -> "CompositeLens":
        left = self.components if isinstance(self, CompositeLens) else [self]
        right = other.components if isinstance(other, CompositeLens) else [other]
        return CompositeLens(left + right)


@dataclass
class CompositeLens(Lens):
    """Überlagerung mehrerer Linsen in derselben Ebene."""

    components: list[Lens] = field(default_factory=list)

    def deflection(self, x, y):
        ax = np.zeros(np.broadcast(x, y).shape)
        ay = np.zeros_like(ax)
        for comp in self.components:
            dx, dy = comp.deflection(x, y)
            ax = ax + dx
            ay = ay + dy
        return ax, ay

    def convergence(self, x, y):
        k = np.zeros(np.broadcast(x, y).shape)
        for comp in self.components:
            k = k + comp.convergence(x, y)
        return k


@dataclass
class PointMass(Lens):
    """Punktmasse (Stern, Schwarzes Loch): alpha = theta_E^2 / |theta|."""

    theta_E: float
    x0: float = 0.0
    y0: float = 0.0

    @classmethod
    def from_mass(cls, geom: LensGeometry, mass_msun: float, **kw) -> "PointMass":
        return cls(theta_E=geom.theta_E_point_mass(mass_msun), **kw)

    def deflection(self, x, y):
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        r2 = dx**2 + dy**2 + _EPS
        f = self.theta_E**2 / r2
        return f * dx, f * dy

    def convergence(self, x, y):
        return np.zeros(np.broadcast(x, y).shape)


@dataclass
class SIS(Lens):
    """Singuläre isotherme Sphäre: flache Rotationskurve, |alpha| = theta_E."""

    theta_E: float
    x0: float = 0.0
    y0: float = 0.0

    @classmethod
    def from_sigma_v(cls, geom: LensGeometry, sigma_v_kms: float, **kw) -> "SIS":
        return cls(theta_E=geom.theta_E_sis(sigma_v_kms), **kw)

    def deflection(self, x, y):
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        r = np.sqrt(dx**2 + dy**2) + _EPS
        return self.theta_E * dx / r, self.theta_E * dy / r

    def convergence(self, x, y):
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        r = np.sqrt(dx**2 + dy**2) + _EPS
        return self.theta_E / (2 * r)


@dataclass
class SIE(Lens):
    """Singulärer isothermer Ellipsoid (Keeton 2001, mit optionalem Kernradius s).

    q    Achsenverhältnis (0 < q <= 1), große Achse unter Winkel phi (rad).
    theta_E ist im Zwischenachsen-Konvention definiert: für q -> 1 geht das
    Modell in die SIS mit gleichem theta_E über.
    """

    theta_E: float
    q: float = 0.8
    phi: float = 0.0
    s: float = 0.0
    x0: float = 0.0
    y0: float = 0.0

    def _b(self):
        return self.theta_E * np.sqrt(self.q)

    def deflection(self, x, y):
        q = min(max(self.q, 1e-3), 1.0)
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        xr, yr = _rotate(dx, dy, self.phi)
        b, s = self._b(), self.s
        if q > 1 - 1e-6:
            r = np.sqrt(xr**2 + yr**2 + s**2) + _EPS
            # für q=1 und s=0 ist das exakt die SIS
            f = b * (r - s) / (r**2 - s**2 + _EPS)
            ax, ay = f * xr, f * yr
        else:
            psi = np.sqrt(q**2 * (s**2 + xr**2) + yr**2) + _EPS
            e = np.sqrt(1 - q**2)
            pre = b / e
            ax = pre * np.arctan(e * xr / (psi + s))
            ay = pre * np.arctanh(np.clip(e * yr / (psi + q**2 * s), -1 + 1e-15, 1 - 1e-15))
        return _rotate_back(ax, ay, self.phi)

    def convergence(self, x, y):
        q = min(max(self.q, 1e-3), 1.0)
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        xr, yr = _rotate(dx, dy, self.phi)
        return self._b() / (2 * np.sqrt(q**2 * (self.s**2 + xr**2) + yr**2) + _EPS)


@dataclass
class ExternalShear(Lens):
    """Konstante externe Scherung (Gezeitenfeld der Umgebung)."""

    gamma1: float = 0.0
    gamma2: float = 0.0

    @classmethod
    def from_polar(cls, gamma: float, phi: float) -> "ExternalShear":
        return cls(gamma * np.cos(2 * phi), gamma * np.sin(2 * phi))

    def deflection(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return self.gamma1 * x + self.gamma2 * y, self.gamma2 * x - self.gamma1 * y

    def convergence(self, x, y):
        return np.zeros(np.broadcast(x, y).shape)


def _nfw_F(x):
    """Hilfsfunktion F(x) der projizierten NFW-Dichte."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    lo = x < 1 - 1e-6
    hi = x > 1 + 1e-6
    out[lo] = np.arccosh(1 / x[lo]) / np.sqrt(1 - x[lo] ** 2)
    out[hi] = np.arccos(1 / x[hi]) / np.sqrt(x[hi] ** 2 - 1)
    return out


def _nfw_h(x):
    """h(x) = ln(x/2) + Integralterm; projizierte Masse in Zylinder ~ h(x)."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, 1.0 + np.log(0.5))
    lo = x < 1 - 1e-6
    hi = x > 1 + 1e-6
    xl, xh = x[lo], x[hi]
    out[lo] = np.log(xl / 2) + 2 / np.sqrt(1 - xl**2) * np.arctanh(np.sqrt((1 - xl) / (1 + xl)))
    out[hi] = np.log(xh / 2) + 2 / np.sqrt(xh**2 - 1) * np.arctan(np.sqrt((xh - 1) / (xh + 1)))
    return out


@dataclass
class NFW(Lens):
    """Navarro-Frenk-White-Halo -- das Standardprofil Dunkler-Materie-Halos.

    kappa_s  = rho_s r_s / Sigma_crit (dimensionslos)
    theta_s  = Skalenradius in arcsec
    """

    kappa_s: float
    theta_s: float
    x0: float = 0.0
    y0: float = 0.0

    @classmethod
    def from_m200(cls, geom: LensGeometry, m200_msun: float, concentration: float, **kw) -> "NFW":
        """Halo aus M200 (Masse innerhalb r200, mittlere Dichte 200 rho_crit(z_d)) und c."""
        from astropy import units as u

        rho_c = geom.cosmo.critical_density(geom.z_d).to(u.M_sun / u.kpc**3).value
        r200 = (3 * m200_msun / (800 * np.pi * rho_c)) ** (1 / 3)  # kpc
        c = concentration
        r_s = r200 / c
        delta_c = 200 / 3 * c**3 / (np.log(1 + c) - c / (1 + c))
        rho_s = delta_c * rho_c  # M_sun/kpc^3
        sigma_crit_kpc2 = geom.sigma_crit.value * 1e6  # M_sun/kpc^2
        kappa_s = rho_s * r_s / sigma_crit_kpc2
        theta_s = r_s / geom.kpc_per_arcsec
        return cls(kappa_s=kappa_s, theta_s=theta_s, **kw)

    def _xr(self, x, y):
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        r = np.sqrt(dx**2 + dy**2) + _EPS
        return dx, dy, r, r / self.theta_s

    def deflection(self, x, y):
        dx, dy, r, xs = self._xr(x, y)
        amp = 4 * self.kappa_s * self.theta_s * _nfw_h(xs) / xs
        return amp * dx / r, amp * dy / r

    def convergence(self, x, y):
        _, _, _, xs = self._xr(x, y)
        xs = np.where(np.abs(xs - 1) < 1e-4, 1 + 1e-4, xs)
        return 2 * self.kappa_s * (1 - _nfw_F(xs)) / (xs**2 - 1)


def einstein_radius(lens: Lens, r_max: float = 1e3) -> float:
    """Einsteinradius einer (näherungsweise) kreissymmetrischen Linse.

    Löst |alpha(theta)| = theta entlang der x-Achse, d.h. mittlere Konvergenz
    innerhalb theta gleich 1. Gibt NaN zurück, wenn die Linse subkritisch ist.
    """
    from scipy.optimize import brentq

    def g(r):
        ax, ay = lens.deflection(r, 0.0)
        return float(np.hypot(ax, ay)) - r

    r_min = 1e-6
    if g(r_min) <= 0:
        return float("nan")
    r_hi = r_min
    while g(r_hi) > 0:
        r_hi *= 2
        if r_hi > r_max:
            return float("nan")
    return float(brentq(g, r_hi / 2, r_hi, xtol=1e-10))
