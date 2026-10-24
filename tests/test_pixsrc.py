import numpy as np
import pytest
from scipy.signal import fftconvolve

from lenscape.imagefit import gaussian_psf
from lenscape.lenses import SIE, ExternalShear
from lenscape.pixsrc import SourceGrid, SourceInverter, fit_lens_by_evidence, mapping_matrix, psf_matrix

PIX, N, SS = 0.06, 80, 2


def clumpy_source(x, y):
    out = 0.0
    for a, x0, y0, s in [(1.0, 0.05, 0.02, 0.06), (0.7, -0.08, 0.07, 0.04), (0.5, 0.12, -0.1, 0.05)]:
        out = out + a * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * s * s))
    return out


def mock(lens, sigma=0.02, seed=0):
    fine = PIX / SS
    c = (np.arange(N * SS) - (N * SS - 1) / 2) * fine
    X, Y = np.meshgrid(c, c)
    bx, by = lens.ray_shoot(X, Y)
    img = clumpy_source(bx, by).reshape(N, SS, N, SS).mean(axis=(1, 3))
    psf = gaussian_psf(0.1, PIX)
    img = fftconvolve(img, psf, mode="same")
    rng = np.random.default_rng(seed)
    return img + rng.normal(0, sigma, img.shape), psf


def ring_mask(r_in=0.6, r_out=2.0):
    yy, xx = (np.mgrid[0:N, 0:N] - (N - 1) / 2) * PIX
    rr = np.hypot(xx, yy)
    return (rr > r_in) & (rr < r_out)


def test_psf_matrix_matches_fft_convolution():
    rng = np.random.default_rng(1)
    img = rng.random((30, 30))
    psf = gaussian_psf(0.1, PIX)
    B = psf_matrix(img.shape, psf)
    np.testing.assert_allclose((B @ img.ravel()).reshape(img.shape), fftconvolve(img, psf, mode="same"), atol=1e-12)


def test_mapping_rows_sum_to_one_inside_grid():
    lens = SIE(1.2, q=0.8)
    L = mapping_matrix(lens.ray_shoot, (N, N), PIX, SourceGrid(30, 2.5), SS)
    sums = np.asarray(L.sum(axis=1)).ravel()
    assert np.allclose(sums[sums > 0.999], 1.0)


def test_reconstructs_clumpy_source_and_finds_einstein_radius():
    truth = SIE(1.4, q=0.8, phi=0.6) + ExternalShear(0.03, -0.01)
    data, psf = mock(truth)
    g = SourceGrid(30, 0.45)
    inv = SourceInverter(data, np.full(data.shape, 0.02), ring_mask(), PIX, psf, g, SS)
    res = inv.invert(truth.ray_shoot)
    gx, gy = g.axes
    GX, GY = np.meshgrid(gx, gy)
    assert np.corrcoef(clumpy_source(GX, GY).ravel(), res.source.ravel())[0, 1] > 0.97
    assert 0.7 < res.reduced_chi2 < 1.1

    make = lambda p: SIE(p["thetaE"], q=0.8, phi=0.6) + ExternalShear(0.03, -0.01)  # noqa: E731
    p, _ = fit_lens_by_evidence(inv, make, {"thetaE": 1.37}, ["thetaE"], scales={"thetaE": 0.02}, n_rounds=1)
    assert p["thetaE"] == pytest.approx(1.4, abs=0.005)
