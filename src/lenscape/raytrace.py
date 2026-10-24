"""Ray-Tracing: Bildebene -> Quellebene und Rendern gelinster Bilder."""

from __future__ import annotations

import numpy as np

from .lenses import Lens
from .sources import Source


def make_grid(half_width: float, n: int):
    """Quadratisches Pixelgitter (Pixelmitten) von -half_width bis +half_width arcsec."""
    pix = 2 * half_width / n
    c = (np.arange(n) + 0.5) * pix - half_width
    x, y = np.meshgrid(c, c)
    return x, y


def render(lens: Lens, source: Source, half_width: float = 3.0, n: int = 300, supersample: int = 2):
    """Gelinstes Bild der Quelle; Oberflächenhelligkeit bleibt erhalten (Liouville).

    Jedes Pixel wird in ``supersample``^2 Unterstrahlen zerlegt, um Kanten
    an kritischen Kurven zu glätten.
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
