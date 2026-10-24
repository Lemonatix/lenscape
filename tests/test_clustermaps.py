import numpy as np
import pytest

from lenscape.clustermaps import MassMap, deflection_from_kappa, effective_einstein_radius, enclosed_mass_profile, sigma_crit_inf
from lenscape.cosmology import LensGeometry, make_cosmology
from lenscape.lenses import einstein_radius

PIX = 0.5
N = 401
c = (np.arange(N) - (N - 1) / 2) * PIX
X, Y = np.meshgrid(c, c)
K0, S = 3.0, 12.0  # Gauß-Konvergenz, Breite in arcsec
KAPPA = K0 * np.exp(-(X**2 + Y**2) / (2 * S**2))


def alpha_gauss(r):
    """Analytisch: alpha(r) = 2 K0 S^2 (1 - exp(-r^2 / 2S^2)) / r."""
    return 2 * K0 * S**2 * (1 - np.exp(-(r**2) / (2 * S**2))) / r


def test_fft_deflection_matches_analytic():
    ax, ay = deflection_from_kappa(KAPPA, PIX)
    r = np.hypot(X, Y)
    m = (r > 2) & (r < 60)
    np.testing.assert_allclose(np.hypot(ax, ay)[m], alpha_gauss(r[m]), rtol=2e-3)
    # Richtung: radial nach außen
    assert np.all(ax[m] * X[m] + ay[m] * Y[m] > 0)


def test_massmap_is_a_lens_and_scales_with_source_redshift():
    m = MassMap(KAPPA, PIX)
    th_inf = einstein_radius(m, r_max=80)
    # analytisch: mittleres kappa in theta_E = 1 -> 2 K0 S^2 (1 - e^{-t^2/2S^2}) = t^2
    from scipy.optimize import brentq
    expected = brentq(lambda t: 2 * K0 * S**2 * (1 - np.exp(-t**2 / (2 * S**2))) - t**2, 1, 80)
    assert th_inf == pytest.approx(expected, rel=2e-3)

    geom = LensGeometry(0.3, 1.0)
    m1 = m.at_source_redshift(geom)
    assert m1.ratio == pytest.approx((geom.D_ds / geom.D_s).value)
    assert einstein_radius(m1, r_max=80) < th_inf  # nähere Quelle -> kleinerer Ring


def test_effective_einstein_radius_from_critical_curve():
    m = MassMap(KAPPA, PIX)
    assert effective_einstein_radius(m) == pytest.approx(einstein_radius(m, r_max=80), rel=1e-2)


def test_enclosed_mass_converges_to_total():
    cosmo = make_cosmology()
    z_d = 0.3
    m = MassMap(KAPPA, PIX)
    from astropy import units as u
    kpc = (cosmo.angular_diameter_distance(z_d) * (np.pi / 180 / 3600)).to(u.kpc).value
    total = K0 * 2 * np.pi * (S * kpc) ** 2 * sigma_crit_inf(z_d, cosmo).value
    big = enclosed_mass_profile(m, z_d, cosmo, [80 * kpc])[0]
    assert big == pytest.approx(total, rel=1e-3)
