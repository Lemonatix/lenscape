"""Ray tracing: image plane -> source plane, and rendering of lensed images."""

from __future__ import annotations

import numpy as np

from .lenses import Lens
from .sources import Source


def make_grid(half_width: float, n: int):
    """Square pixel grid (pixel centres) from -half_width to +half_width arcsec."""
    pix = 2 * half_width / n
    c = (np.arange(n) + 0.5) * pix - half_width
    x, y = np.meshgrid(c, c)
    return x, y


def render(lens: Lens, source: Source, half_width: float = 3.0, n: int = 300, supersample: int = 2):
    """Lensed image of the source; surface brightness is conserved (Liouville).

    Each pixel is split into ``supersample``^2 sub-rays to smooth edges near
    critical curves.
    """
    pix = 2 * half_width / n
    x, y = make_grid(half_width, n)
    if supersample > 1:
        offs = ((np.arange(supersample) + 0.5) / supersample - 0.5) * pix
        acc = None
        for ox in offs:
            for oy in offs:
                bx, by = lens.ray_shoot(x + ox, y + oy)
                val = source.brightness(bx, by)
                acc = val if acc is None else acc + val
        return acc / supersample**2
    bx, by = lens.ray_shoot(x, y)
    return source.brightness(bx, by)


def render_unlensed(source: Source, half_width: float = 3.0, n: int = 300):
    x, y = make_grid(half_width, n)
    return source.brightness(x, y)


def magnification_map(lens: Lens, half_width: float = 3.0, n: int = 300):
    x, y = make_grid(half_width, n)
    return lens.magnification(x, y)


def convergence_map(lens: Lens, half_width: float = 3.0, n: int = 300):
    x, y = make_grid(half_width, n)
    return lens.convergence(x, y)
