"""Zeitverzögerungen zwischen Bildern einer gelinsten Quelle.

Licht entlang verschiedener Bilder legt unterschiedliche Wege zurück und läuft
durch unterschiedlich tiefe Gravitationspotentiale. Die Ankunftszeit eines
Bildes bei theta ist (bis auf eine Konstante)

    t(theta) = D_dt / c * tau(theta),   tau = |theta - beta|^2 / 2 - psi(theta),

mit der Zeitverzögerungs-Distanz D_dt = (1 + z_d) D_d D_s / D_ds  ∝  1/H0.
Misst man die Verzögerungen (Lichtkurven variabler Quasare) und kennt tau aus
einem Linsenmodell, folgt D_dt und damit H0 -- unabhängig von der
Entfernungsleiter und vom frühen Universum.
"""

from __future__ import annotations

import numpy as np
from astropy import constants as const
from astropy import units as u

from .cosmology import LensGeometry, RAD_PER_ARCSEC
from .lenses import Lens

SECONDS_PER_DAY = 86400.0


def fermat_potential(lens: Lens, x, y, beta_x: float, beta_y: float):
    """tau(theta) in arcsec^2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return 0.5 * ((x - beta_x) ** 2 + (y - beta_y) ** 2) - lens.potential(x, y)


def time_delay_days(geom: LensGeometry, delta_tau_arcsec2):
    """Δt in Tagen für eine Differenz des Fermat-Potentials (arcsec^2)."""
    D_dt = geom.time_delay_distance()
    t = D_dt / const.c * np.asarray(delta_tau_arcsec2) * RAD_PER_ARCSEC**2
    return t.to(u.s).value / SECONDS_PER_DAY


def time_delays(lens: Lens, geom: LensGeometry, images, beta_x: float, beta_y: float):
    """Ankunftszeiten (Tage) der Bilder relativ zum zuerst ankommenden.

    ``images``: Array (N, >=2) mit x, y in den ersten Spalten, z. B. aus
    :func:`lenscape.solver.find_images`.
    """
    images = np.atleast_2d(np.asarray(images, dtype=float))
    tau = fermat_potential(lens, images[:, 0], images[:, 1], beta_x, beta_y)
    return time_delay_days(geom, tau - tau.min())
