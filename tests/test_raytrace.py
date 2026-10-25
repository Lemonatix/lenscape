import numpy as np
import pytest

from lenscape.caustics import caustics, critical_curves
from lenscape.lenses import SIE, SIS, ExternalShear
from lenscape.raytrace import render, render_unlensed
from lenscape.solver import find_images
from lenscape.sources import GaussianSource, SersicSource


def test_sis_two_images_and_magnifications():
    thE, beta = 1.0, 0.3
    imgs = find_images(SIS(thE), beta, 0.0)
    assert len(imgs) == 2
    xs = np.sort(imgs[:, 0])
    np.testing.assert_allclose(xs, [beta - thE, beta + thE], atol=1e-6)
    mu = {round(x, 6): m for x, m, in zip(imgs[:, 0], imgs[:, 2])}
    assert mu[round(beta + thE, 6)] == pytest.approx(1 + thE / beta, rel=1e-3)
    assert mu[round(beta - thE, 6)] == pytest.approx(1 - thE / beta, rel=1e-3)


def test_sis_single_image_outside_einstein_radius():
    assert len(find_images(SIS(1.0), 1.5, 0.0)) == 1


def test_sie_with_shear_can_make_four_images():
    lens = SIE(1.0, q=0.7) + ExternalShear(0.05, 0.0)
    imgs = find_images(lens, 0.05, 0.03)
    assert len(imgs) == 4  # the singular SIE has no central image
    assert (np.abs(imgs[:, 2]) > 1).all()


def test_sis_critical_curve_is_einstein_ring():
    curves = critical_curves(SIS(1.2), half_width=2.0, n=300)
    r = np.concatenate([np.hypot(c[:, 0], c[:, 1]) for c in curves])
    assert np.median(r) == pytest.approx(1.2, rel=1e-2)


def test_sie_caustic_exists():
    cs = caustics(SIE(1.0, q=0.6), half_width=2.0, n=300)
    assert len(cs) >= 1


def test_lensing_magnifies_total_flux():
    src = GaussianSource(0.1, 0.0, sigma=0.05)
    lensed = render(SIS(1.0), src, half_width=3.0, n=300).sum()
    unlensed = render_unlensed(src, half_width=3.0, n=300).sum()
    assert lensed > 2 * unlensed


def test_sersic_peak_at_center():
    s = SersicSource(R_e=0.3, n=4)
    assert s.brightness(0.0, 0.0) > s.brightness(0.3, 0.0)
