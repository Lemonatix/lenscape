import numpy as np
import pytest

from lenscape.cosmology import LensGeometry, make_cosmology
from lenscape.distance_likelihoods import KDE1D, KDE2D, SkewLogNormal
from lenscape.h0licow import H0LICOW_DIR, TimeDelayLens, load_h0licow
from lenscape.h0_inference import _distances, infer_h0


def test_skew_lognormal_is_normalized():
    d = SkewLogNormal(mu=7.0, sigma=0.2, lam=4000.0)
    x = np.linspace(4000.01, 20000, 200001)
    assert np.trapezoid(np.exp(d.logpdf(x)), x) == pytest.approx(1.0, rel=1e-3)
    assert d.logpdf(np.array([3999.0]))[0] == -np.inf


def test_kde1d_recovers_gaussian():
    rng = np.random.default_rng(0)
    k = KDE1D(rng.normal(5000, 300, 200_000), bandwidth=20)
    x = np.linspace(3000, 7000, 4001)
    p = np.exp(k.logpdf(x))
    assert np.trapezoid(p, x) == pytest.approx(1.0, rel=1e-3)
    assert x[np.argmax(p)] == pytest.approx(5000, abs=30)


def test_kde2d_peak_near_mean():
    rng = np.random.default_rng(1)
    dd = rng.normal(1000, 100, 200_000)
    ddt = rng.normal(3000, 200, 200_000)
    k = KDE2D(dd, ddt, bandwidth=20)
    assert k.logpdf(1000, 3000) > k.logpdf(1200, 3000) > k.logpdf(1400, 3600)


def test_distances_match_astropy():
    lens = TimeDelayLens("test", 0.5, 1.8, "", lambda dd, ddt: 0.0)
    dd_u, ddt_u = _distances(lens, 0.3, -1.0)
    g = LensGeometry(0.5, 1.8, make_cosmology(H0=70, Om0=0.3, w0=-1.0))
    c_over_h0 = 299792.458 / 70
    assert dd_u * c_over_h0 == pytest.approx(g.D_d.value, rel=1e-5)
    assert ddt_u * c_over_h0 == pytest.approx(g.time_delay_distance().value, rel=1e-5)


def test_synthetic_lens_recovers_h0():
    """A lens with a narrow D_dt likelihood around the H0=70 value must return H0≈70."""
    lens0 = TimeDelayLens("x", 0.5, 2.0, "", lambda dd, ddt: 0.0)
    _, ddt_u = _distances(lens0, 0.3, -1.0)
    ddt_true = ddt_u * 299792.458 / 70.0
    lens = TimeDelayLens("x", 0.5, 2.0, "", lambda dd, ddt: -0.5 * ((ddt - ddt_true) / (0.02 * ddt_true)) ** 2)
    med, lo, hi = infer_h0([lens], Om=np.array([0.3])).summary()
    assert med == pytest.approx(70.0, abs=0.2)
    assert lo == pytest.approx(1.4, abs=0.2) and hi == pytest.approx(1.4, abs=0.2)


@pytest.mark.skipif(not (H0LICOW_DIR / "J1206_final.csv").exists(),
                    reason="H0LiCOW data missing (python scripts/fetch_h0licow.py)")
def test_reproduces_wong2020_flat_lcdm():
    """Wong et al. (2020): H0 = 73.3 +1.7 -1.8 km/s/Mpc (six lenses, flat ΛCDM)."""
    med, lo, hi = infer_h0(load_h0licow()).summary()
    assert med == pytest.approx(73.3, abs=0.3)
    assert lo == pytest.approx(1.8, abs=0.25)
    assert hi == pytest.approx(1.7, abs=0.25)
