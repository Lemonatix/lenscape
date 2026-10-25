"""Critical curves (det A = 0 in the image plane) and caustics (their image in the source plane).

When a source crosses a caustic, the number of images changes by two --
this is where bright arcs form.
"""

from __future__ import annotations

import contourpy
import numpy as np

from .lenses import Lens
from .raytrace import make_grid


def critical_curves(lens: Lens, half_width: float = 3.0, n: int = 400):
    """List of (N, 2) arrays with points on the critical curves (arcsec)."""
    x, y = make_grid(half_width, n)
    det = lens.jacobian_det(x, y)
    gen = contourpy.contour_generator(x, y, det)
    return [line for line in gen.lines(0.0) if len(line) > 2]


def caustics(lens: Lens, half_width: float = 3.0, n: int = 400, curves=None):
    """Caustics = critical curves mapped through the lens equation."""
    if curves is None:
        curves = critical_curves(lens, half_width, n)
    out = []
    for c in curves:
        bx, by = lens.ray_shoot(c[:, 0], c[:, 1])
        out.append(np.column_stack([bx, by]))
    return out
