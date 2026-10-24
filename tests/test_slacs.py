"""Tests gegen die veröffentlichten SLACS-Werte (Bolton+2008, Auger+2009)."""

import numpy as np
import pytest

from lenscape.catalogs import DATA_DIR, SLACS_REF_COSMO, load_slacs
from lenscape.cosmology import make_cosmology
from lenscape.strong_lenses import einstein_quantities

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "slacs_bolton2008" / "table5.dat").exists(),
    reason="SLACS-Daten fehlen (python scripts/fetch_cds.py)",
)


@pytest.fixture(scope="module")
def slacs():
    return einstein_quantities(load_slacs(), make_cosmology(**SLACS_REF_COSMO))


def test_sample_size(slacs):
    assert len(slacs) == 63
    assert np.all(slacs["z_s"] > slacs["z_d"])


def test_einstein_masses_match_auger2009(slacs):
    """Unsere M_E müssen die publizierten log M_E (gleiche Kosmologie) reproduzieren."""
    m = np.isfinite(slacs["logM_E_auger"])
    assert m.sum() > 55
    dlog = np.log10(slacs["M_E"][m]) - slacs["logM_E_auger"][m]
    assert np.abs(dlog).max() < 0.015  # Tabelle ist auf 0.01 dex gerundet


def test_einstein_radii_kpc_match_auger2009(slacs):
    m = np.isfinite(slacs["R_E_kpc_auger"])
    np.testing.assert_allclose(slacs["R_E_kpc"][m], slacs["R_E_kpc_auger"][m], atol=0.01)


def test_stellar_fraction_consistent_with_auger2009(slacs):
    ours = 1 - np.asarray(slacs["fDM_chab"])
    theirs = np.asarray(slacs["fstar_chab"])
    m = np.isfinite(ours) & np.isfinite(theirs)
    assert np.corrcoef(ours[m], theirs[m])[0, 1] > 0.9
    assert np.median(ours[m] - theirs[m]) == pytest.approx(0.0, abs=0.03)


def test_isothermal_lens_velocity_dispersion_near_unity(slacs):
    """sigma_SDSS / sigma_SIE ~ 1: Massenprofile sind nahezu isotherm."""
    f = slacs["f_SIE"][slacs["good_sigma"]]
    assert 0.85 < np.mean(f) < 1.15


def test_cosmology_changes_masses(slacs):
    other = einstein_quantities(load_slacs(), make_cosmology(H0=70, Om0=0.3, w0=-0.5))
    assert not np.allclose(other["M_E"], slacs["M_E"], rtol=1e-3)
