"""Linsenmodell direkt an ein Bild fitten (Vorwärtsmodellierung).

Modell eines Pixels:

    I(θ) = PSF * [ A_L · S_L(θ)  +  A_S · S_S(β(θ)) ] + b

* S_L: Sérsic-Profil der Linsengalaxie (Licht)
* S_S: Sérsic-Profil der Quelle, ausgewertet an β = θ − α(θ) (SIE + externe Scherung)
* A_L, A_S, b: Helligkeiten und Himmelshintergrund -- werden für jeden Satz
  nichtlinearer Parameter *linear* gelöst (gewichtete kleinste Quadrate)
* PSF: Gauß (ACS/WFC F814W: FWHM ≈ 0,1″), Faltung auf einem feineren Gitter

Elliptizitäten werden als (e1, e2) mit q = (1−|e|)/(1+|e|), φ = ½·atan2(e2, e1)
parametrisiert, damit es keine Winkelsprünge gibt.

Gefittet wird in drei Stufen (wie in der Praxis üblich):
1. nur Linsenlicht, Ringbereich maskiert
2. Masse + Quelle bei festgehaltenem Lichtprofil, mehrere Startpunkte
3. alles gemeinsam
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.special import gammaincinv

from .lenses import SIE, ExternalShear
from .sources import _elliptical_radius


# ---------------------------------------------------------------- Hilfsfunktionen
def q_phi_from_e(e1: float, e2: float) -> tuple[float, float]:
    e = min(np.hypot(e1, e2), 0.95)
    return (1 - e) / (1 + e), 0.5 * np.arctan2(e2, e1)


def e_from_q_phi(q: float, phi: float) -> tuple[float, float]:
    e = (1 - q) / (1 + q)
    return e * np.cos(2 * phi), e * np.sin(2 * phi)


def sersic_unit(x, y, x0, y0, R_e, n, e1, e2):
    """Sérsic-Profil mit Amplitude 1 bei R_e."""
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
    # Linsenlicht
    "xl", "yl", "Rl", "nl", "e1l", "e2l",
    # Masse (Zentrum = Lichtzentrum)
    "thetaE", "e1m", "e2m", "g1", "g2",
    # Quelle
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
    residual: np.ndarray            # (Daten − Modell) / σ
    history: list = field(default_factory=list)

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / max(self.dof, 1)

    @property
    def mass_q_phi(self) -> tuple[float, float]:
        return q_phi_from_e(self.params["e1m"], self.params["e2m"])


class LensImageModel:
    """Vorwärtsmodell auf einem quadratischen Ausschnitt mit Pixelgröße ``pixscale``."""

    def __init__(self, shape: tuple[int, int], pixscale: float, psf_fwhm: float = 0.1, supersample: int = 2):
        self.shape = shape
        self.pixscale = pixscale
        self.ss = supersample
        ny, nx = shape
        fine = pixscale / supersample
        xs = (np.arange(nx * supersample) - (nx * supersample - 1) / 2) * fine
        ys = (np.arange(ny * supersample) - (ny * supersample - 1) / 2) * fine
        self.X, self.Y = np.meshgrid(xs, ys)
        self.psf = gaussian_psf(psf_fwhm, fine)

    # ---------------------------------------------------------------- Bausteine
    def _bin(self, img):
        s = self.ss
        ny, nx = self.shape
        return img.reshape(ny, s, nx, s).mean(axis=(1, 3))

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
        """Amplituden (A_L, A_S, b) per gewichteten kleinsten Quadraten."""
        L, S = self.components(p)
        cols = [L.ravel(), S.ravel(), np.ones(L.size)] if use_source else [L.ravel(), np.ones(L.size)]
        A = np.column_stack(cols) * w.ravel()[:, None]
        coef, *_ = np.linalg.lstsq(A, data.ravel() * w.ravel(), rcond=None)
        if use_source:
            aL, aS, b = coef
        else:
            (aL, b), aS = coef, 0.0
        aL, aS = max(aL, 0.0), max(aS, 0.0)  # physikalisch: keine negativen Helligkeiten
        return aL * L, aS * S, b, {"lens_light": aL, "source": aS, "background": b}


def _vec(p: dict, names: list[str]) -> np.ndarray:
    return np.array([p[k] for k in names])


def _bounds(names, p0):
    lo = np.array([BOUNDS[k][0] for k in names])
    hi = np.array([BOUNDS[k][1] for k in names])
    # Positionsgrenzen relativ zum Startwert der Lichtmitte
    return lo, hi


def fit_lens_image(data: np.ndarray, sigma: np.ndarray, pixscale: float, theta_E0: float = 1.2,
                   q0: float = 0.8, phi0: float = 0.0, psf_fwhm: float = 0.1, supersample: int = 2,
                   n_starts: int = 8, seed: int = 0, verbose: bool = False) -> FitResult:
    """Dreistufiger Fit (siehe Moduldokumentation). Koordinaten relativ zur Bildmitte."""
    model = LensImageModel(data.shape, pixscale, psf_fwhm, supersample)
    w = 1.0 / np.asarray(sigma, float)
    rng = np.random.default_rng(seed)
    hist = []

    p = {k: 0.0 for k in PARAMS}
    e1, e2 = e_from_q_phi(q0, phi0)
    p.update(Rl=1.0, nl=4.0, thetaE=theta_E0, e1m=e1, e2m=e2, Rs=0.15, ns=1.0)

    def residual_fn(names, mask=None, use_source=True):
        def f(v):
            q = dict(p)
            q.update(zip(names, v))
            L, S, b, _ = model.solve_linear(q, data, w if mask is None else w * mask, use_source)
            r = (data - L - S - b) * w
            return (r if mask is None else r * mask).ravel()
        return f

    # ---- Stufe 1: Linsenlicht, Ring maskiert
    ny, nx = data.shape
    yy, xx = (np.mgrid[0:ny, 0:nx] - np.array([(ny - 1) / 2, (nx - 1) / 2])[:, None, None]) * pixscale
    rr = np.hypot(xx, yy)
    ring = (rr > 0.5 * theta_E0) & (rr < 1.7 * theta_E0)
    mask = (~ring).astype(float)
    lo, hi = _bounds(LIGHT, p)
    sol = least_squares(residual_fn(LIGHT, mask, use_source=False), _vec(p, LIGHT), bounds=(lo, hi), x_scale="jac")
    p.update(zip(LIGHT, sol.x))
    hist.append(("Licht", 2 * sol.cost))
    if verbose:
        print("Stufe 1 (Licht):", {k: round(p[k], 3) for k in LIGHT})

    # ---- Stufe 2: Masse + Quelle, mehrere Starts
    lo, hi = _bounds(MASS_SOURCE, p)
    best = None
    for k in range(n_starts):
        start = dict(p)
        if k > 0:
            ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0.0, 0.4) * theta_E0
            start.update(xs=rad * np.cos(ang), ys=rad * np.sin(ang),
                         thetaE=theta_E0 * rng.uniform(0.85, 1.15), Rs=rng.uniform(0.05, 0.4))
        v0 = np.clip(_vec(start, MASS_SOURCE), lo + 1e-6, hi - 1e-6)
        sol = least_squares(residual_fn(MASS_SOURCE), v0, bounds=(lo, hi), x_scale="jac", max_nfev=400)
        if best is None or sol.cost < best.cost:
            best = sol
        if verbose:
            print(f"Stufe 2, Start {k}: chi2 = {2 * sol.cost:.1f}, thetaE = {sol.x[0]:.3f}")
    p.update(zip(MASS_SOURCE, best.x))
    hist.append(("Masse+Quelle", 2 * best.cost))

    # ---- Stufe 3: alles gemeinsam
    lo, hi = _bounds(PARAMS, p)
    sol = least_squares(residual_fn(PARAMS), np.clip(_vec(p, PARAMS), lo + 1e-6, hi - 1e-6),
                        bounds=(lo, hi), x_scale="jac", max_nfev=600)
    p.update(zip(PARAMS, sol.x))
    hist.append(("gemeinsam", 2 * sol.cost))

    L, S, b, amps = model.solve_linear(p, data, w)
    m = L + S + b
    chi2 = float(np.sum(((data - m) * w) ** 2))
    return FitResult(p, amps, chi2, data.size - len(PARAMS) - 3, m, L, S, (data - m) * w, hist)


def estimate_noise(data: np.ndarray, exptime: float | None = None, border: int = 10) -> np.ndarray:
    """σ pro Pixel: Hintergrundrauschen (robust aus dem Rand) plus Poisson-Anteil (bei e⁻/s-Bildern)."""
    edge = np.concatenate([data[:border].ravel(), data[-border:].ravel(),
                           data[:, :border].ravel(), data[:, -border:].ravel()])
    med = np.median(edge)
    sig_bkg = 1.4826 * np.median(np.abs(edge - med))
    var = np.full(data.shape, sig_bkg**2)
    if exptime and exptime > 0:
        var += np.clip(data - med, 0, None) / exptime
    return np.sqrt(var)
