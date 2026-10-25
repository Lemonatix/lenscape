"""Find the image positions of a point source.

Method (Schneider, Ehlers & Falco; "triangle mapping"):
1. Tile the image plane with triangles and map their corners to the source plane.
2. Flag every triangle whose image contains the source position beta.
3. Refine each candidate with a Newton solver onto beta.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import root

from .lenses import Lens
from .raytrace import make_grid


def _in_triangle(px, py, ax, ay, bx, by, cx, cy):
    d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
    d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def find_images(lens: Lens, beta_x: float, beta_y: float, half_width: float = 4.0,
                n: int = 400, tol: float = 1e-10):
    """All images (x, y, magnification) of the point source at (beta_x, beta_y)."""
    x, y = make_grid(half_width, n)
    bx, by = lens.ray_shoot(x, y)

    cands = []
    # two triangles per grid cell
    for (i0, j0), (i1, j1), (i2, j2) in (
        ((0, 0), (0, 1), (1, 0)),
        ((1, 1), (1, 0), (0, 1)),
    ):
        sl = lambda a, i, j: a[i : a.shape[0] - 1 + i, j : a.shape[1] - 1 + j]
        hit = _in_triangle(
            beta_x, beta_y,
            sl(bx, i0, j0), sl(by, i0, j0),
            sl(bx, i1, j1), sl(by, i1, j1),
            sl(bx, i2, j2), sl(by, i2, j2),
        )
        for r, c in zip(*np.nonzero(hit)):
            cands.append((
                (x[r + i0, c + j0] + x[r + i1, c + j1] + x[r + i2, c + j2]) / 3,
                (y[r + i0, c + j0] + y[r + i1, c + j1] + y[r + i2, c + j2]) / 3,
            ))

    def f(p):
        sx, sy = lens.ray_shoot(p[0], p[1])
        return [float(sx) - beta_x, float(sy) - beta_y]

    pix = 2 * half_width / n
    images: list[tuple[float, float]] = []
    for cx, cy in cands:
        sol = root(f, [cx, cy], tol=tol)
        if not sol.success or np.hypot(*sol.fun) > 1e-6:
            continue
        px, py = sol.x
        if any(np.hypot(px - qx, py - qy) < pix for qx, qy in images):
            continue
        images.append((px, py))

    out = np.array([(px, py, float(lens.magnification(px, py))) for px, py in images])
    return out.reshape(-1, 3)
