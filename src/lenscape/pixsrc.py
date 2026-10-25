"""Pixelated source reconstruction with regularization (Bayesian semi-linear inversion).

Instead of describing the source by a smooth profile, it is reconstructed
freely on a pixel grid. For a fixed lens model the image is linear in the
source pixels s:

    d = M s + n,      M = B · L

* L  maps source pixels to image pixels (ray tracing + bilinear weights)
* B  convolves with the PSF
* n  noise with covariance C (diagonal, σ²)

Because the problem is ill-posed, a smoothness prior λ · ½ sᵀ H s is added
(gradient or curvature regularization). The most probable solution

    s_MP = A⁻¹ Mᵀ C⁻¹ d,      A = Mᵀ C⁻¹ M + λ H

and the regularization strength λ follow from the Bayesian evidence (Suyu,
Marshall, Hobson & Blandford 2006, MNRAS 371, 983; Warren & Dye 2003):

    ln P(d|λ) = −E_D − λ E_S − ½ ln det A + ½ N_s ln λ + ½ ln det H + const.

The evidence also ranks different lens models: a better model explains the
data with a simpler (smoother) source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar

from .lenses import Lens


@dataclass
class SourceGrid:
    """Square pixel grid in the source plane (arcsec)."""

    n: int = 40
    half: float = 0.6
    x0: float = 0.0
    y0: float = 0.0

    @property
    def step(self) -> float:
        return 2 * self.half / (self.n - 1)

    @property
    def axes(self):
        c = np.linspace(-self.half, self.half, self.n)
        return c + self.x0, c + self.y0

    @property
    def size(self) -> int:
        return self.n * self.n


def _image_coords(shape, pixscale, supersample):
    ny, nx = shape
    fine = pixscale / supersample
    xs = (np.arange(nx * supersample) - (nx * supersample - 1) / 2) * fine
    ys = (np.arange(ny * supersample) - (ny * supersample - 1) / 2) * fine
    return np.meshgrid(xs, ys)


def mapping_matrix(lens_fn, shape, pixscale: float, grid: SourceGrid, supersample: int = 2) -> sp.csr_matrix:
    """L: (N_image × N_source). ``lens_fn(x, y) -> (βx, βy)`` is the ray-tracing function.

    Each image pixel is split into supersample² sub-rays; each one hits the
    source plane and distributes its weight bilinearly onto the four
    surrounding source pixels. Rays outside the grid contribute nothing.
    """
    X, Y = _image_coords(shape, pixscale, supersample)
    bx, by = lens_fn(X, Y)
    n, h = grid.n, grid.step
    u = (bx - (grid.x0 - grid.half)) / h
    v = (by - (grid.y0 - grid.half)) / h
    i0 = np.floor(u).astype(int)
    j0 = np.floor(v).astype(int)
    fu, fv = u - i0, v - j0
    ny, nx = shape
    s = supersample
    # index of the coarse image pixel of each sub-ray
    rows_fine = (np.arange(ny * s)[:, None] // s) * nx + (np.arange(nx * s)[None, :] // s)
    rows, cols, vals = [], [], []
    for di, dj, wgt in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
        ii, jj = i0 + di, j0 + dj
        ok = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n) & (wgt > 0)
        rows.append(rows_fine[ok])
        cols.append((jj * n + ii)[ok])
        vals.append(wgt[ok] / s**2)
    L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ny * nx, grid.size))
    return L.tocsr()


def psf_matrix(shape, psf: np.ndarray) -> sp.csr_matrix:
    """B: convolution with the (small) PSF kernel as a sparse matrix (zeros outside)."""
    ny, nx = shape
    kh, kw = psf.shape
    cy, cx = kh // 2, kw // 2
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, vals = [], [], []
    for a in range(kh):
        for b in range(kw):
            w = psf[a, b]
            if w < 1e-6 * psf.max():
                continue
            dy, dx = a - cy, b - cx
            ys = slice(max(0, dy), min(ny, ny + dy))
            xs = slice(max(0, dx), min(nx, nx + dx))
            ys_src = slice(max(0, -dy), min(ny, ny - dy))
            xs_src = slice(max(0, -dx), min(nx, nx - dx))
            rows.append(idx[ys, xs].ravel())
            cols.append(idx[ys_src, xs_src].ravel())
            vals.append(np.full(rows[-1].size, w))
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(ny * nx, ny * nx)).tocsr()


def regularization_matrix(grid: SourceGrid, kind: str = "gradient") -> sp.csr_matrix:
    """H = Dᵀ D with difference operator D and zero boundary (which makes it positive definite)."""
    n = grid.n
    I = sp.identity(n, format="csr")
    if kind == "gradient":
        d1 = sp.diags([np.ones(n), -np.ones(n - 1)], [0, 1], shape=(n, n))  # last row: boundary
        Dx = sp.kron(I, d1)
        Dy = sp.kron(d1, I)
        return (Dx.T @ Dx + Dy.T @ Dy).tocsr()
    if kind == "curvature":
        d2 = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], shape=(n, n))
        D = sp.kron(I, d2) + sp.kron(d2, I)
        return (D.T @ D).tocsr()
    raise ValueError(kind)


@dataclass
class Inversion:
    source: np.ndarray          # (n, n) reconstructed source
    model: np.ndarray           # image model (source part only), full image size
    lam: float
    log_evidence: float
    chi2: float
    n_data: int
    grid: SourceGrid

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / max(self.n_data, 1)


class SourceInverter:
    """Holds data, mask, PSF and regularization; inverts for arbitrary lens models."""

    def __init__(self, data, sigma, mask, pixscale: float, psf: np.ndarray, grid: SourceGrid,
                 supersample: int = 2, reg: str = "gradient"):
        self.shape = data.shape
        self.pixscale = pixscale
        self.grid = grid
        self.ss = supersample
        self.sel = (mask & np.isfinite(sigma) & np.isfinite(data)).ravel()
        self.d = np.asarray(data, float).ravel()[self.sel]
        self.w = 1.0 / np.asarray(sigma, float).ravel()[self.sel]  # 1/σ
        self.B = psf_matrix(self.shape, psf)
        self.H = regularization_matrix(grid, reg)
        Hd = self.H.toarray()
        self.logdetH = 2 * np.sum(np.log(np.diag(cho_factor(Hd, lower=True)[0])))

    def _system(self, lens_fn):
        L = mapping_matrix(lens_fn, self.shape, self.pixscale, self.grid, self.ss)
        M_full = self.B @ L
        M = M_full[self.sel]
        Mw = sp.diags(self.w) @ M
        F = (Mw.T @ Mw).toarray()           # Mᵀ C⁻¹ M
        b = Mw.T @ (self.d * self.w)        # Mᵀ C⁻¹ d
        return M_full, M, F, b

    def _solve(self, F, b, M, lam):
        A = F + lam * self.H.toarray()
        try:
            c = cho_factor(A, lower=True)
        except np.linalg.LinAlgError:
            return None
        s = cho_solve(c, b)
        r = (M @ s - self.d) * self.w
        ED = 0.5 * float(r @ r)
        ES = 0.5 * float(s @ (self.H @ s))
        logdetA = 2 * np.sum(np.log(np.diag(c[0])))
        logE = (-ED - lam * ES - 0.5 * logdetA + 0.5 * self.grid.size * np.log(lam) + 0.5 * self.logdetH
                - 0.5 * len(self.d) * np.log(2 * np.pi) + np.sum(np.log(self.w)))
        return s, logE, 2 * ED

    def invert(self, lens_fn, lam: float | None = None, lam_bounds=(1e-3, 1e4)) -> Inversion:
        """Reconstruct the source; without ``lam``, λ is optimized via the evidence."""
        M_full, M, F, b = self._system(lens_fn)
        if lam is None:
            def neg(loglam):
                out = self._solve(F, b, M, 10**loglam)
                return np.inf if out is None else -out[1]
            res = minimize_scalar(neg, bounds=np.log10(lam_bounds), method="bounded", options={"xatol": 0.02})
            lam = 10**res.x
        s, logE, chi2 = self._solve(F, b, M, lam)
        model = (M_full @ s).reshape(self.shape)
        return Inversion(s.reshape(self.grid.n, self.grid.n), model, lam, logE, chi2, len(self.d), self.grid)


def fit_lens_by_evidence(inverter: SourceInverter, make_lens, p0: dict[str, float], free: list[str],
                         scales: dict[str, float] | None = None, n_rounds: int = 2, maxiter: int = 400,
                         verbose: bool = False):
    """Choose lens parameters that maximize the Bayesian evidence.

    ``make_lens(params) -> Lens``. Alternates: optimize λ for a fixed model,
    then the model parameters for fixed λ (Nelder-Mead). Returns (params, Inversion).
    """
    from scipy.optimize import minimize

    p = dict(p0)
    scales = scales or {}
    sc = np.array([scales.get(k, 0.05) for k in free])
    inv = inverter.invert(make_lens(p).ray_shoot)
    for rnd in range(n_rounds):
        lam = inv.lam

        def neg(v):
            q = dict(p)
            q.update(zip(free, np.asarray(v) * sc))
            try:
                out = inverter.invert(make_lens(q).ray_shoot, lam=lam)
            except Exception:  # noqa: BLE001
                return np.inf
            return -out.log_evidence

        v0 = np.array([p[k] for k in free]) / sc
        res = minimize(neg, v0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 0.05,
                                "initial_simplex": v0 + np.vstack([np.zeros(len(free)), 0.5 * np.eye(len(free))])})
        p.update(zip(free, res.x * sc))
        inv = inverter.invert(make_lens(p).ray_shoot)
        if verbose:
            print(f"  round {rnd + 1}: ln E = {inv.log_evidence:.1f}, λ = {inv.lam:.3g}, "
                  + ", ".join(f"{k} = {p[k]:.4f}" for k in free))
    return p, inv
