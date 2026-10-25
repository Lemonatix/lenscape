import numpy as np
import pytest

from lenscape.cosmology import LensGeometry, make_cosmology


def test_flat_default():
    c = make_cosmology()
    assert c.Ok0 == pytest.approx(0.0, abs=1e-4)


def test_requires_ordered_redshifts():
    with pytest.raises(ValueError):
        LensGeometry(1.0, 0.5)


def test_distances_positive_and_not_additive():
    g = LensGeometry(0.3, 1.0)
    assert g.D_d.value > 0 and g.D_s.value > 0 and g.D_ds.value > 0
    # angular diameter distances do not add up
    assert g.D_d.value + g.D_ds.value > g.D_s.value


def test_sigma_crit_order_of_magnitude():
    # typical galaxy-scale lens: a few 1000 M_sun/pc^2
    g = LensGeometry(0.3, 1.0)
    assert 1e3 < g.sigma_crit.value < 1e4


def test_sis_roundtrip():
    g = LensGeometry(0.2, 0.7)
    th = g.theta_E_sis(240.0)
    assert g.sigma_v_from_theta_E(th) == pytest.approx(240.0, rel=1e-10)


def test_point_mass_mass_inside_einstein_radius():
    # for a point mass, M(<theta_E) must equal its mass exactly
    g = LensGeometry(0.5, 2.0)
    M = 1e12
    th = g.theta_E_point_mass(M)
    assert g.mass_in_theta_E(th) == pytest.approx(M, rel=1e-8)


def test_dark_energy_changes_einstein_radius():
    lcdm = LensGeometry(0.5, 2.0, make_cosmology(w0=-1.0))
    phantom = LensGeometry(0.5, 2.0, make_cosmology(w0=-1.5))
    assert lcdm.theta_E_sis(250) != pytest.approx(phantom.theta_E_sis(250), rel=1e-4)
