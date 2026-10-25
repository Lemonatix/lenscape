"""Time delays between the images of a lensed source.

Light reaching us along different images travels paths of different length
and through gravitational potentials of different depth. The arrival time of
an image at theta is (up to a constant)

    t(theta) = D_dt / c * tau(theta),   tau = |theta - beta|^2 / 2 - psi(theta),

with the time-delay distance D_dt = (1 + z_d) D_d D_s / D_ds  ∝  1/H0.
Measuring the delays (light curves of variable quasars) and knowing tau from
a lens model gives D_dt and thus H0 -- independent of the distance ladder and
of the early universe.
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
    """Δt in days for a Fermat-potential difference (arcsec^2)."""
    D_dt = geom.time_delay_distance()
    t = D_dt / const.c * np.asarray(delta_tau_arcsec2) * RAD_PER_ARCSEC**2
    return t.to(u.s).value / SECONDS_PER_DAY


def time_delays(lens: Lens, geom: LensGeometry, images, beta_x: float, beta_y: float):
    """Arrival times (days) of the images relative to the first one to arrive.

    ``images``: array (N, >=2) with x, y in the first columns, e.g. from
    :func:`lenscape.solver.find_images`.
    """
    images = np.atleast_2d(np.asarray(images, dtype=float))
    tau = fermat_potential(lens, images[:, 0], images[:, 1], beta_x, beta_y)
    return time_delay_days(geom, tau - tau.min())
