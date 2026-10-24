import numpy as np
import pytest

from lenscape.cosmology import LensGeometry
from lenscape.lenses import NFW, SIE, SIS, ExternalShear, PointMass

rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, 300)
Y = rng.uniform(-3, 3, 300)
MASK = X**2 + Y**2 > 0.05

MODELS = [
    SIS(1.1, x0=0.2),
    SIE(1.2, q=0.6, phi=0.4),
    SIE(1.0, q=0.75, phi=-1.0, s=0.15),
    SIE(1.0, q=1.0, s=0.3),
    NFW(0.3, 2.0, y0=-0.1),
    PointMass(1.0),
    ExternalShear(0.05, -0.03),
]


@pytest.mark.parametrize("lens", MODELS, ids=lambda m: type(m).__name__)
def test_poisson_equation(lens):
    """div(alpha) = 2 kappa"""
    axx, axy, ayx, ayy = lens.hessian(X, Y, h=1e-5)
    np.testing.assert_allclose(((axx + ayy) / 2)[MASK], lens.convergence(X, Y)[MASK], atol=1e-6)


@pytest.mark.parametrize("lens", MODELS, ids=lambda m: type(m).__name__)
def test_deflection_is_curl_free(lens):
    """alpha ist ein Gradientenfeld -> symmetrische Jacobi-Matrix."""
    _, axy, ayx, _ = lens.hessian(X, Y, h=1e-5)
    np.testing.assert_allclose(axy[MASK], ayx[MASK], atol=1e-6)


def test_sie_tends_to_sis():
    a = SIE(1.3, q=0.99999).deflection(X, Y)
    b = SIS(1.3).deflection(X, Y)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_sis_constant_deflection():
    ax, ay = SIS(1.7).deflection(X, Y)
    np.testing.assert_allclose(np.hypot(ax, ay), 1.7, rtol=1e-9)


def test_einstein_ring_maps_to_origin():
    for lens in (SIS(1.3), PointMass(1.3)):
        t = np.linspace(0, 2 * np.pi, 50)
        bx, by = lens.ray_shoot(1.3 * np.cos(t), 1.3 * np.sin(t))
        np.testing.assert_allclose(np.hypot(bx, by), 0, atol=1e-9)


def test_nfw_mean_convergence_matches_deflection():
    g = LensGeometry(0.3, 1.0)
    halo = NFW.from_m200(g, 1e15, 5.0)
    th = 2 * halo.theta_s
    r = np.linspace(1e-5, th, 40001)
    k = halo.convergence(r, 0 * r)
    mean_kappa = np.trapezoid(2 * np.pi * r * k, r) / (np.pi * th**2)
    assert halo.deflection(th, 0.0)[0] / th == pytest.approx(mean_kappa, rel=1e-4)


def test_composite_adds_up():
    a, b = SIS(1.0), ExternalShear(0.1, 0.0)
    total = (a + b).deflection(X, Y)
    np.testing.assert_allclose(total[0], a.deflection(X, Y)[0] + b.deflection(X, Y)[0])


def test_einstein_radius_numeric():
    from lenscape.lenses import einstein_radius

    assert einstein_radius(SIS(1.4)) == pytest.approx(1.4, rel=1e-8)
    assert einstein_radius(PointMass(0.8)) == pytest.approx(0.8, rel=1e-8)
    assert np.isnan(einstein_radius(NFW(0.01, 5.0)))  # subkritischer Halo
