import numpy as np
import pytest

from lenscape.cosmology import LensGeometry
from lenscape.lenses import NFW, SIE, SIS, ExternalShear, PointMass
from lenscape.solver import find_images
from lenscape.timedelay import fermat_potential, time_delay_days, time_delays

rng = np.random.default_rng(7)
X = rng.uniform(-3, 3, 200)
Y = rng.uniform(-3, 3, 200)
MASK = X**2 + Y**2 > 0.05
H = 1e-5

MODELS = [
    PointMass(1.0),
    SIS(1.2, x0=0.1),
    SIE(1.1, q=0.6, phi=0.4),
    SIE(1.0, q=0.7, s=0.2, phi=1.0),
    SIE(1.0, q=1.0, s=0.3),
    NFW(0.3, 2.0),
    ExternalShear(0.05, -0.02),
    SIE(1.0, q=0.7) + ExternalShear(0.05, 0.01),
]


@pytest.mark.parametrize("lens", MODELS, ids=lambda m: type(m).__name__)
def test_potential_gradient_is_deflection(lens):
    gx = (lens.potential(X + H, Y) - lens.potential(X - H, Y)) / (2 * H)
    gy = (lens.potential(X, Y + H) - lens.potential(X, Y - H)) / (2 * H)
    ax, ay = lens.deflection(X, Y)
    np.testing.assert_allclose(gx[MASK], ax[MASK], atol=1e-6)
    np.testing.assert_allclose(gy[MASK], ay[MASK], atol=1e-6)


def test_images_are_stationary_points_of_fermat_potential():
    lens = SIE(1.0, q=0.7) + ExternalShear(0.05, 0.0)
    bx, by = 0.05, 0.03
    for x, y, _ in find_images(lens, bx, by):
        gx = (fermat_potential(lens, x + H, y, bx, by) - fermat_potential(lens, x - H, y, bx, by)) / (2 * H)
        gy = (fermat_potential(lens, x, y + H, bx, by) - fermat_potential(lens, x, y - H, bx, by)) / (2 * H)
        assert abs(gx) < 1e-5 and abs(gy) < 1e-5


def test_sis_time_delay_analytic():
    """SIS: Δt = 2 D_dt beta theta_E / c."""
    geom = LensGeometry(0.5, 2.0)
    thE, beta = 1.0, 0.3
    imgs = find_images(SIS(thE), beta, 0.0)
    dt = time_delays(SIS(thE), geom, imgs, beta, 0.0)
    expected = time_delay_days(geom, 2 * beta * thE)
    assert dt.max() == pytest.approx(expected, rel=1e-6)
    # the image outside the Einstein ring arrives first
    assert imgs[np.argmin(dt), 0] > 0


def test_typical_delays_are_days_to_months():
    geom = LensGeometry(0.5, 2.0)
    lens = SIE(1.2, q=0.7) + ExternalShear(0.05, 0.0)
    imgs = find_images(lens, 0.05, 0.03)
    dt = time_delays(lens, geom, imgs, 0.05, 0.03)
    assert 1 < dt.max() < 200
