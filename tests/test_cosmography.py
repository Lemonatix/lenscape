import numpy as np
import pytest

from lenscape.cosmography import fit_flat_wcdm, model_distance_ratio, observed_distance_ratio
from lenscape.cosmology import LensGeometry, make_cosmology


def test_distance_ratio_matches_astropy():
    for Om, w in [(0.3, -1.0), (0.2, -1.5), (0.5, -0.6)]:
        g = LensGeometry(0.35, 1.2, make_cosmology(H0=70, Om0=Om, w0=w))
        expected = (g.D_ds / g.D_s).value
        # astropy berücksichtigt Strahlung nicht (Tcmb0=0), daher exakt vergleichbar
        assert model_distance_ratio(0.35, 1.2, Om, w) == pytest.approx(expected, rel=1e-5)


def test_observed_ratio_roundtrip():
    g = LensGeometry(0.2, 0.6)
    th = g.theta_E_sis(260.0)
    assert observed_distance_ratio(th, 260.0) == pytest.approx((g.D_ds / g.D_s).value, rel=1e-10)


def test_fit_recovers_truth_on_synthetic_sample():
    rng = np.random.default_rng(3)
    n = 400
    z_d = rng.uniform(0.1, 0.8, n)
    z_s = z_d + rng.uniform(0.2, 2.5, n)
    sigma = rng.uniform(180, 340, n)
    truth = dict(Om=0.3, w=-1.0, f_E=1.02)
    ratio = model_distance_ratio(z_d, z_s, truth["Om"], truth["w"])
    theta = 4 * np.pi * (truth["f_E"] * sigma / 299792.458) ** 2 * ratio / (np.pi / 180 / 3600)
    theta *= np.exp(rng.normal(0, 0.03, n))
    e_sigma = 0.01 * sigma
    res = fit_flat_wcdm(theta, sigma, e_sigma, z_d, z_s, e_theta_frac=0.02,
                        Om=np.linspace(0.1, 0.6, 26), w=np.linspace(-2.0, -0.4, 33),
                        sigma_int=np.linspace(0, 0.1, 11))
    best = res.best
    assert best["f_E"] == pytest.approx(1.02, abs=0.02)
    # Om und w sind entartet, aber die Wahrheit muss im 2-sigma-Bereich liegen
    j = np.argmin(np.abs(res.w - truth["w"]))
    i = np.argmin(np.abs(res.Om - truth["Om"]))
    assert res.delta_chi2()[j, i] < 6.17
