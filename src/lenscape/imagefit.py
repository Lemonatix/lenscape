"""Fit a lens model directly to an image (forward modelling).

Model of one pixel:

    I(θ) = PSF * [ A_L · S_L(θ)  +  A_S · S_S(β(θ)) ] + b

* S_L: Sérsic profile of the lens galaxy (light)
* S_S: Sérsic profile of the source, evaluated at β = θ − α(θ) (SIE + external shear)
* A_L, A_S, b: amplitudes and sky background -- solved *linearly* for every set
  of non-linear parameters (weighted least squares)
* PSF: Gaussian (ACS/WFC F814W: FWHM ≈ 0.1″), convolution on a finer grid

Ellipticities are parametrized as (e1, e2) with q = (1−|e|)/(1+|e|),
φ = ½·atan2(e2, e1), so that there are no angle wraps.

The fit proceeds in three stages (as is common practice):
1. lens light only, ring region masked
2. mass + source with the light profile held fixed, several starting points
3. everything jointly
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.special import gammaincinv

from .lenses import SIE, ExternalShear
from .sources import _elliptical_radius


# ---------------------------------------------------------------- helpers
def q_phi_from_e(e1: float, e2: float) -> tuple[float, float]:
    e = min(np.hypot(e1, e2), 0.95)
    return (1 - e) / (1 + e), 0.5 * np.arctan2(e2, e1)


def e_from_q_phi(q: float, phi: float) -> tuple[float, float]:
    e = (1 - q) / (1 + q)
    return e * np.cos(2 * phi), e * np.sin(2 * phi)


def sersic_unit(x, y, x0, y0, R_e, n, e1, e2):
    """Sérsic profile with amplitude 1 at R_e."""
    q, phi = q_phi_from_e(e1, e2)
    r = _elliptical_radius(x, y, x0, y0, q, phi)
    b_n = gammaincinv(2 * n, 0.5)
    return np.exp(-b_n * ((np.maximum(r, 1e-6) / R_e) ** (1 / n) - 1))


def gaussian_psf(fwhm_arcsec: float, pixscale: float, half_size: int | None = None) -> np.ndarray:
    sigma = fwhm_arcsec / 2.3548 / pixscale
    h = half_size or int(np.ceil(4 * sigma))
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


# ---------------------------------------------------------------- Parameter
PARAMS = [
    # lens light
    "xl", "yl", "Rl", "nl", "e1l", "e2l",
    # mass (centre = light centre)
    "thetaE", "e1m", "e2m", "g1", "g2",
    # source
    "xs", "ys", "Rs", "ns", "e1s", "e2s",
]
LIGHT = PARAMS[:6]
MASS_SOURCE = PARAMS[6:]

BOUNDS = {
    "xl": (-1.0, 1.0), "yl": (-1.0, 1.0), "Rl": (0.05, 10.0), "nl": (0.5, 8.0),
    "e1l": (-0.7, 0.7), "e2l": (-0.7, 0.7),
    "thetaE": (0.2, 4.0), "e1m": (-0.6, 0.6), "e2m": (-0.6, 0.6), "g1": (-0.2, 0.2), "g2": (-0.2, 0.2),
    "xs": (-1.5, 1.5), "ys": (-1.5, 1.5), "Rs": (0.01, 1.5), "ns": (0.3, 4.0),
    "e1s": (-0.7, 0.7), "e2s": (-0.7, 0.7),
}


@dataclass
class FitResult:
    params: dict[str, float]
    amplitudes: dict[str, float]
    chi2: float
    dof: int
    model: np.ndarray
    lens_light: np.ndarray
    lensed_source: np.ndarray
    residual: np.ndarray            # (data − model) / σ
    history: list = field(default_factory=list)

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / max(self.dof, 1)

    @property
    def mass_q_phi(self) -> tuple[float, float]:
        return q_phi_from_e(self.params["e1m"], self.params["e2m"])


class LensImageModel:
    """Forward model on a square cutout with pixel size ``pixscale``."""

    def __init__(self, shape: tuple[int, int], pixscale: float, psf_fwhm: float = 0.1, supersample: int = 2):
        self.shape = shape
        self.pixscale = pixscale
        self.ss = supersample
        ny, nx = shape
        fine = pixscale / supersample
        self.psf = gaussian_psf(psf_fwhm, fine)
        # border in whole image pixels so that the convolution also picks up light from outside
        self.pad = int(np.ceil((self.psf.shape[0] // 2) / supersample)) + 1
        NY, NX = ny + 2 * self.pad, nx + 2 * self.pad
        xs = (np.arange(NX * supersample) - (NX * supersample - 1) / 2) * fine
        ys = (np.arange(NY * supersample) - (NY * supersample - 1) / 2) * fine
        self.X, self.Y = np.meshgrid(xs, ys)

    # ---------------------------------------------------------------- building blocks
    def _bin(self, img):
        s, p = self.ss, self.pad
        ny, nx = self.shape
        NY, NX = ny + 2 * p, nx + 2 * p
        return img.reshape(NY, s, NX, s).mean(axis=(1, 3))[p:p + ny, p:p + nx]

    def _convolve(self, img):
        return self._bin(fftconvolve(img, self.psf, mode="same"))

    def lens(self, p) -> SIE:
        q, phi = q_phi_from_e(p["e1m"], p["e2m"])
        return SIE(p["thetaE"], q=q, phi=phi, x0=p["xl"], y0=p["yl"]) + ExternalShear(p["g1"], p["g2"])

    def components(self, p) -> tuple[np.ndarray, np.ndarray]:
        light = sersic_unit(self.X, self.Y, p["xl"], p["yl"], p["Rl"], p["nl"], p["e1l"], p["e2l"])
        bx, by = self.lens(p).ray_shoot(self.X, self.Y)
        src = sersic_unit(bx, by, p["xs"], p["ys"], p["Rs"], p["ns"], p["e1s"], p["e2s"])
        return self._convolve(light), self._convolve(src)

    def solve_linear(self, p, data, w, use_source=True):
        """Amplitudes (A_L, A_S, b) via weighted least squares."""
        L, S = self.components(p)
        cols = [L.ravel(), S.ravel(), np.ones(L.size)] if use_source else [L.ravel(), np.ones(L.size)]
        A = np.column_stack(cols) * w.ravel()[:, None]
        coef, *_ = np.linalg.lstsq(A, data.ravel() * w.ravel(), rcond=None)
        if use_source:
            aL, aS, b = coef
        else:
            (aL, b), aS = coef, 0.0
        aL, aS = max(aL, 0.0), max(aS, 0.0)  # physical: no negative brightness
        return aL * L, aS * S, b, {"lens_light": aL, "source": aS, "background": b}


def _vec(p: dict, names: list[str]) -> np.ndarray:
    return np.array([p[k] for k in names])


def _bounds(names, p0):
    lo = np.array([BOUNDS[k][0] for k in names])
    hi = np.array([BOUNDS[k][1] for k in names])
    # position bounds relative to the initial light centre
    return lo, hi


def fit_lens_image(data: np.ndarray, sigma: np.ndarray, pixscale: float, theta_E0: float = 1.2,
                   q0: float = 0.8, phi0: float = 0.0, psf_fwhm: float = 0.1, supersample: int = 2,
                   n_starts: int = 8, seed: int = 0, verbose: bool = False,
                   fixed: dict[str, float] | None = None) -> FitResult:
    """Three-stage fit (see module docs). Coordinates relative to the image centre.

    Pixels with ``sigma = inf`` (or NaN data) are ignored, e.g. cosmic rays.
    ``fixed`` holds parameters fixed, e.g. ``{"g1": 0, "g2": 0}`` for a pure SIE without shear.
    """
    fixed = dict(fixed or {})
    free = lambda names: [k for k in names if k not in fixed]  # noqa: E731
    LIGHT_, MS_, ALL_ = free(LIGHT), free(MASS_SOURCE), free(PARAMS)
    model = LensImageModel(data.shape, pixscale, psf_fwhm, supersample)
    sigma = np.asarray(sigma, float)
    bad = ~np.isfinite(data) | ~np.isfinite(sigma) | (sigma <= 0)
    data = np.where(bad, 0.0, data)
    w = np.where(bad, 0.0, 1.0 / np.where(bad, 1.0, sigma))
    rng = np.random.default_rng(seed)
    hist = []

    p = {k: 0.0 for k in PARAMS}
    e1, e2 = e_from_q_phi(q0, phi0)
    p.update(Rl=1.0, nl=4.0, thetaE=theta_E0, e1m=e1, e2m=e2, Rs=0.15, ns=1.0)
    p.update(fixed)

    def residual_fn(names, mask=None, use_source=True):
        def f(v):
            q = dict(p)
            q.update(zip(names, v))
            L, S, b, _ = model.solve_linear(q, data, w if mask is None else w * mask, use_source)
            r = (data - L - S - b) * w
            return (r if mask is None else r * mask).ravel()
        return f

    # ---- stage 1: lens light, ring masked
    ny, nx = data.shape
    yy, xx = (np.mgrid[0:ny, 0:nx] - np.array([(ny - 1) / 2, (nx - 1) / 2])[:, None, None]) * pixscale
    rr = np.hypot(xx, yy)
    ring = (rr > 0.5 * theta_E0) & (rr < 1.7 * theta_E0)
    mask = (~ring).astype(float)
    lo, hi = _bounds(LIGHT_, p)
    sol = least_squares(residual_fn(LIGHT_, mask, use_source=False), _vec(p, LIGHT_), bounds=(lo, hi), x_scale="jac")
    p.update(zip(LIGHT_, sol.x))
    hist.append(("Licht", 2 * sol.cost))
    if verbose:
        print("stage 1 (light):", {k: round(p[k], 3) for k in LIGHT_})

    # ---- stage 2: mass + source, several starts
    lo, hi = _bounds(MS_, p)
    best = None
    for k in range(n_starts):
        start = dict(p)
        if k > 0:
            ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0.0, 0.4) * theta_E0
            start.update(xs=rad * np.cos(ang), ys=rad * np.sin(ang),
                         thetaE=theta_E0 * rng.uniform(0.85, 1.15), Rs=rng.uniform(0.05, 0.4))
        v0 = np.clip(_vec(start, MS_), lo + 1e-6, hi - 1e-6)
        sol = least_squares(residual_fn(MS_), v0, bounds=(lo, hi), x_scale="jac", max_nfev=400)
        if best is None or sol.cost < best.cost:
            best = sol
        if verbose:
            print(f"stage 2, start {k}: chi2 = {2 * sol.cost:.1f}, thetaE = {sol.x[MS_.index("thetaE")]:.3f}")
    p.update(zip(MS_, best.x))
    hist.append(("Masse+Quelle", 2 * best.cost))

    # ---- stage 3: everything jointly
    lo, hi = _bounds(ALL_, p)
    sol = least_squares(residual_fn(ALL_), np.clip(_vec(p, ALL_), lo + 1e-6, hi - 1e-6),
                        bounds=(lo, hi), x_scale="jac", max_nfev=600)
    p.update(zip(ALL_, sol.x))
    hist.append(("gemeinsam", 2 * sol.cost))

    L, S, b, amps = model.solve_linear(p, data, w)
    m = L + S + b
    chi2 = float(np.sum(((data - m) * w) ** 2))
    return FitResult(p, amps, chi2, int((w > 0).sum()) - len(ALL_) - 3, m, L, S, (data - m) * w, hist)


def estimate_noise(data: np.ndarray, exptime: float | None = None, border: int = 10) -> np.ndarray:
    """σ per pixel: background noise plus Poisson term (for e⁻/s images).

    The background noise is estimated robustly (MAD) from the image border after
    subtracting a smooth trend (median filter, 9 px) -- otherwise the outskirts
    of the galaxy would inflate the noise estimate.
    """
    from scipy.ndimage import median_filter

    hp = data - median_filter(data, size=9, mode="reflect")
    edge = np.concatenate([hp[:border].ravel(), hp[-border:].ravel(),
                           hp[:, :border].ravel(), hp[:, -border:].ravel()])
    # variance of the high-pass ≈ σ² (1 + 1/81) for white noise
    sig_bkg = 1.4826 * np.median(np.abs(edge - np.median(edge))) / np.sqrt(1 + 1 / 81)
    var = np.full(data.shape, sig_bkg**2)
    if exptime and exptime > 0:
        base = np.median(data[:border])
        var += np.clip(data - base, 0, None) / exptime
    return np.sqrt(var)
