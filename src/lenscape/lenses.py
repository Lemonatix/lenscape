"""Thin-lens mass models.

All angles are in arcseconds. Every model returns the *scaled* deflection
angle alpha(theta) = D_ds/D_s * alpha_hat, so that the lens equation is simply

    beta = theta - alpha(theta)

The convergence kappa = Sigma / Sigma_crit satisfies div(alpha) = 2 kappa, and
the lensing potential psi (arcsec^2) satisfies grad(psi) = alpha. psi gives the
Fermat potential and hence the time delay between images.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cosmology import LensGeometry

_EPS = 1e-12


def _rotate(x, y, phi):
    """Rotate coordinates by -phi (into the principal-axis frame)."""
    c, s = np.cos(phi), np.sin(phi)
    return c * x + s * y, -s * x + c * y


def _rotate_back(ax, ay, phi):
    c, s = np.cos(phi), np.sin(phi)
    return c * ax - s * ay, s * ax + c * ay


class Lens:
    """Base class. Subclasses implement ``deflection`` and ``convergence``."""

    def deflection(self, x, y):
        raise NotImplementedError

    def convergence(self, x, y):
        raise NotImplementedError

    def potential(self, x, y):
        """Lensing potential psi in arcsec^2 (up to a constant)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement a potential")

    # --------------------------------------------------------- derived quantities
    def hessian(self, x, y, h: float = 1e-4):
        """Jacobian of alpha via central finite differences.

        Returns (a_xx, a_xy, a_yx, a_yy)
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
        """det A with A = d beta / d theta = 1 - d alpha / d theta."""
        a_xx, a_xy, a_yx, a_yy = self.hessian(x, y)
        return (1 - a_xx) * (1 - a_yy) - a_xy * a_yx

    def magnification(self, x, y):
        det = self.jacobian_det(x, y)
        return 1.0 / np.where(np.abs(det) < _EPS, np.sign(det) * _EPS + _EPS, det)

    def ray_shoot(self, x, y):
        """Image plane -> source plane (lens equation)."""
        ax, ay = self.deflection(x, y)
        return np.asarray(x) - ax, np.asarray(y) - ay

    def __add__(self, other: "Lens") -> "CompositeLens":
        left = self.components if isinstance(self, CompositeLens) else [self]
        right = other.components if isinstance(other, CompositeLens) else [other]
        return CompositeLens(left + right)


@dataclass
class CompositeLens(Lens):
    """Superposition of several lenses in the same plane."""

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

    def potential(self, x, y):
        p = np.zeros(np.broadcast(x, y).shape)
        for comp in self.components:
            p = p + comp.potential(x, y)
        return p


@dataclass
class PointMass(Lens):
    """Point mass (star, black hole): alpha = theta_E^2 / |theta|."""

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

    def potential(self, x, y):
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        return 0.5 * self.theta_E**2 * np.log(dx**2 + dy**2 + _EPS)


@dataclass
class SIS(Lens):
    """Singular isothermal sphere: flat rotation curve, |alpha| = theta_E."""

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

    def potential(self, x, y):
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        return self.theta_E * np.sqrt(dx**2 + dy**2)


@dataclass
class SIE(Lens):
    """Singular isothermal ellipsoid (Keeton 2001, with optional core radius s).

    q    axis ratio (0 < q <= 1), major axis at angle phi (rad).
    theta_E uses the intermediate-axis convention: for q -> 1 the model
    turns into the SIS with the same theta_E.
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
            # for q=1 and s=0 this is exactly the SIS
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

    def potential(self, x, y):
        """Keeton (2001): psi = x alpha_x + y alpha_y - b s ln[...] (in the principal-axis frame)."""
        q = min(max(self.q, 1e-3), 1.0)
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        xr, yr = _rotate(dx, dy, self.phi)
        ax, ay = _rotate(*self.deflection(x, y), self.phi)
        b, s = self._b(), self.s
        out = xr * ax + yr * ay
        if s > 0:
            if q > 1 - 1e-6:
                R = np.sqrt(xr**2 + yr**2 + s**2)
                return b * (R - s * np.log(s + R))
            psi = np.sqrt(q**2 * (s**2 + xr**2) + yr**2)
            out = out - b * s * np.log(np.sqrt((psi + s) ** 2 + (1 - q**2) * xr**2))
        return out


@dataclass
class ExternalShear(Lens):
    """Constant external shear (tidal field of the environment)."""

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

    def potential(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return 0.5 * self.gamma1 * (x**2 - y**2) + self.gamma2 * x * y


def _nfw_F(x):
    """Auxiliary function F(x) of the projected NFW density."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    lo = x < 1 - 1e-6
    hi = x > 1 + 1e-6
    out[lo] = np.arccosh(1 / x[lo]) / np.sqrt(1 - x[lo] ** 2)
    out[hi] = np.arccos(1 / x[hi]) / np.sqrt(x[hi] ** 2 - 1)
    return out


def _nfw_h(x):
    """h(x) = ln(x/2) + integral term; projected mass in a cylinder ~ h(x)."""
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
    """Navarro-Frenk-White halo -- the standard profile of dark-matter haloes.

    kappa_s  = rho_s r_s / Sigma_crit (dimensionless)
    theta_s  = scale radius in arcsec
    """

    kappa_s: float
    theta_s: float
    x0: float = 0.0
    y0: float = 0.0

    @classmethod
    def from_m200(cls, geom: LensGeometry, m200_msun: float, concentration: float, **kw) -> "NFW":
        """Halo from M200 (mass inside r200, mean density 200 rho_crit(z_d)) and c."""
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

    def potential(self, x, y):
        _, _, _, xs = self._xr(x, y)
        g = np.log(xs / 2) ** 2
        lo = xs < 1
        hi = ~lo
        g = np.where(lo, g - np.arctanh(np.sqrt(np.clip(1 - xs**2, 0, None))) ** 2, g)
        g = np.where(hi, g + np.arctan(np.sqrt(np.clip(xs**2 - 1, 0, None))) ** 2, g)
        return 2 * self.kappa_s * self.theta_s**2 * g


def einstein_radius(lens: Lens, r_max: float = 1e3) -> float:
    """Einstein radius of an (approximately) circularly symmetric lens.

    Solves |alpha(theta)| = theta along the x axis, i.e. mean convergence
    inside theta equal to 1. Returns NaN if the lens is subcritical.
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
