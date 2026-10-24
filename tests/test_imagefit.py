import numpy as np
import pytest

from lenscape.imagefit import PARAMS, LensImageModel, e_from_q_phi, estimate_noise, fit_lens_image, q_phi_from_e


def test_ellipticity_roundtrip():
    for q, phi in [(0.7, 0.3), (0.95, -1.2), (0.5, 1.5)]:
        q2, phi2 = q_phi_from_e(*e_from_q_phi(q, phi))
        assert q2 == pytest.approx(q)
        assert np.cos(2 * (phi2 - phi)) == pytest.approx(1.0)


def test_noise_estimate_recovers_background_sigma():
    rng = np.random.default_rng(0)
    img = rng.normal(0.1, 0.03, (100, 100))
    assert np.median(estimate_noise(img)) == pytest.approx(0.03, rel=0.05)


def test_fit_recovers_synthetic_einstein_ring():
    pix, n = 0.06, 80
    truth = {k: 0.0 for k in PARAMS}
    e1l, e2l = e_from_q_phi(0.8, 0.4)
    e1m, e2m = e_from_q_phi(0.75, 0.5)
    e1s, e2s = e_from_q_phi(0.6, 1.2)
    truth.update(xl=0.03, yl=-0.02, Rl=1.2, nl=4, e1l=e1l, e2l=e2l, thetaE=1.3, e1m=e1m, e2m=e2m,
                 g1=0.03, g2=-0.01, xs=0.08, ys=0.05, Rs=0.12, ns=1.0, e1s=e1s, e2s=e2s)
    m = LensImageModel((n, n), pix, psf_fwhm=0.1, supersample=2)
    L, S = m.components(truth)
    rng = np.random.default_rng(1)
    data = 2.0 * L + 0.8 * S + 0.01 + rng.normal(0, 0.02, (n, n))
    r = fit_lens_image(data, np.full((n, n), 0.02), pix, theta_E0=1.15, q0=0.85, phi0=0.0, n_starts=3)
    assert r.params["thetaE"] == pytest.approx(1.3, rel=0.01)
    assert r.reduced_chi2 == pytest.approx(1.0, abs=0.1)
    q, phi = r.mass_q_phi
    assert q == pytest.approx(0.75, abs=0.03)
