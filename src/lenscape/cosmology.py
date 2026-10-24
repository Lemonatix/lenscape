"""Kosmologie-Schicht: Distanzen und kritische Flächendichte.

Alle Linsenrechnungen hängen nur über drei Winkeldurchmesser-Distanzen von der
Kosmologie ab:

    D_d   Beobachter -> Linse (deflector)
    D_s   Beobachter -> Quelle (source)
    D_ds  Linse -> Quelle

Daraus folgt die kritische Flächendichte

    Sigma_crit = c^2 / (4 pi G) * D_s / (D_d * D_ds),

die jede Massenverteilung in eine dimensionslose Konvergenz kappa = Sigma / Sigma_crit
übersetzt. Über Omega_m, Omega_Lambda und die Zustandsgleichung w der Dunklen
Energie verschieben sich diese Distanzen -- und damit Einsteinradien und
abgeleitete Massen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FLRW, wCDM

ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0
RAD_PER_ARCSEC = 1.0 / ARCSEC_PER_RAD

# Planck 2018 (TT,TE,EE+lowE+lensing+BAO), gerundet
PLANCK18_H0 = 67.66
PLANCK18_OM0 = 0.3111


def make_cosmology(
    H0: float = PLANCK18_H0,
    Om0: float = PLANCK18_OM0,
    Ode0: float | None = None,
    w0: float = -1.0,
) -> FLRW:
    """Erzeuge eine wCDM-Kosmologie.

    Ode0=None bedeutet flaches Universum (Ode0 = 1 - Om0). w0 = -1 ist eine
    kosmologische Konstante (LambdaCDM).
    """
    if Ode0 is None:
        Ode0 = 1.0 - Om0
    return wCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0)


@dataclass
class LensGeometry:
    """Beobachter-Linse-Quelle-Geometrie in einer gegebenen Kosmologie."""

    z_d: float
    z_s: float
    cosmo: FLRW = field(default_factory=make_cosmology)

    def __post_init__(self) -> None:
        if not 0 < self.z_d < self.z_s:
            raise ValueError("Es muss 0 < z_d < z_s gelten.")
        self.D_d = self.cosmo.angular_diameter_distance(self.z_d).to(u.Mpc)
        self.D_s = self.cosmo.angular_diameter_distance(self.z_s).to(u.Mpc)
        self.D_ds = self.cosmo.angular_diameter_distance(self.z_d, self.z_s).to(u.Mpc)

    # ------------------------------------------------------------------ Skalen
    @property
    def sigma_crit(self) -> u.Quantity:
        """Kritische Flächendichte in M_sun / pc^2."""
        sc = const.c**2 / (4 * np.pi * const.G) * self.D_s / (self.D_d * self.D_ds)
        return sc.to(u.M_sun / u.pc**2)

    @property
    def kpc_per_arcsec(self) -> float:
        """Physikalische Länge in der Linsenebene pro Bogensekunde."""
        return (self.D_d * RAD_PER_ARCSEC).to(u.kpc).value

    @property
    def sigma_crit_arcsec2(self) -> float:
        """Kritische Masse pro Quadratbogensekunde in der Linsenebene, in M_sun."""
        area_pc2 = (self.D_d.to(u.pc).value * RAD_PER_ARCSEC) ** 2
        return self.sigma_crit.value * area_pc2

    # ------------------------------------------------------------ Einsteinradien
    def theta_E_point_mass(self, mass_msun: float) -> float:
        """Einsteinradius (arcsec) einer Punktmasse M (in M_sun)."""
        M = mass_msun * u.M_sun
        th2 = 4 * const.G * M / const.c**2 * self.D_ds / (self.D_d * self.D_s)
        return float(np.sqrt(th2.decompose().value) * ARCSEC_PER_RAD)

    def theta_E_sis(self, sigma_v_kms: float) -> float:
        """Einsteinradius (arcsec) einer singulären isothermen Sphäre.

        theta_E = 4 pi (sigma_v / c)^2 * D_ds / D_s
        """
        sv = sigma_v_kms * u.km / u.s
        th = 4 * np.pi * (sv / const.c) ** 2 * self.D_ds / self.D_s
        return float(th.decompose().value * ARCSEC_PER_RAD)

    def sigma_v_from_theta_E(self, theta_E_arcsec: float) -> float:
        """Umkehrung von theta_E_sis: Geschwindigkeitsdispersion in km/s."""
        th = theta_E_arcsec * RAD_PER_ARCSEC
        ratio = (self.D_s / self.D_ds).decompose().value
        return float(np.sqrt(th * ratio / (4 * np.pi)) * const.c.to(u.km / u.s).value)

    def mass_in_theta_E(self, theta_E_arcsec: float) -> float:
        """Projizierte Masse (M_sun) innerhalb des Einsteinradius.

        Unabhängig vom Massenprofil gilt für kreissymmetrische Linsen
        M(<theta_E) = pi * Sigma_crit * (D_d theta_E)^2.
        """
        return float(np.pi * self.sigma_crit_arcsec2 * theta_E_arcsec**2)

    def time_delay_distance(self) -> u.Quantity:
        """D_dt = (1 + z_d) D_d D_s / D_ds -- Basis der H0-Messung aus Zeitverzögerungen."""
        return ((1 + self.z_d) * self.D_d * self.D_s / self.D_ds).to(u.Mpc)

    def summary(self) -> dict[str, float]:
        return {
            "D_d [Mpc]": self.D_d.value,
            "D_s [Mpc]": self.D_s.value,
            "D_ds [Mpc]": self.D_ds.value,
            "Sigma_crit [M_sun/pc^2]": self.sigma_crit.value,
            "kpc/arcsec": self.kpc_per_arcsec,
        }
