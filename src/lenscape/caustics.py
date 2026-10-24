"""Kritische Kurven (det A = 0 in der Bildebene) und Kaustiken (deren Bild in der Quellebene).

Überquert eine Quelle eine Kaustik, ändert sich die Bildanzahl um zwei --
dort entstehen die hellen Bögen.
"""

from __future__ import annotations

import contourpy
import numpy as np

from .lenses import Lens
from .raytrace import make_grid


def critical_curves(lens: Lens, half_width: float = 3.0, n: int = 400):
    """Liste von (N, 2)-Arrays mit Punkten der kritischen Kurven (arcsec)."""
    x, y = make_grid(half_width, n)
    det = lens.jacobian_det(x, y)
    gen = contourpy.contour_generator(x, y, det)
    return [line for line in gen.lines(0.0) if len(line) > 2]


def caustics(lens: Lens, half_width: float = 3.0, n: int = 400, curves=None):
    """Kaustiken = kritische Kurven, durch die Linsengleichung abgebildet."""
    if curves is None:
        curves = critical_curves(lens, half_width, n)
    out = []
    for c in curves:
        bx, by = lens.ray_shoot(c[:, 0], c[:, 1])
        out.append(np.column_stack([bx, by]))
    return out
