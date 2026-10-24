"""Abgleich mit lenstronomy (Birrer & Amara 2018), dem Standardwerkzeug der Community.

Läuft nur, wenn lenstronomy installiert ist: pip install -e ".[validate]"
"""

import numpy as np
import pytest

lenstronomy = pytest.importorskip("lenstronomy")
from lenstronomy.LensModel.lens_model import LensModel  # noqa: E402

from lenscape.imagefit import e_from_q_phi  # noqa: E402
from lenscape.lenses import NFW, SIE, ExternalShear  # noqa: E402

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, 500)
Y = rng.uniform(-3, 3, 500)


def test_sie_and_shear_match_lenstronomy():
    q, phi = 0.7, 0.5
    e1, e2 = e_from_q_phi(q, phi)
    kw = [{"theta_E": 1.3, "e1": e1, "e2": e2, "center_x": 0.1, "center_y": -0.05},
          {"gamma1": 0.04, "gamma2": -0.02}]
    lm = LensModel(["SIE", "SHEAR"])
    ours = SIE(1.3, q=q, phi=phi, x0=0.1, y0=-0.05) + ExternalShear(0.04, -0.02)
    ax, ay = lm.alpha(X, Y, kw)
    bx, by = ours.deflection(X, Y)
    np.testing.assert_allclose(bx, ax, atol=1e-8)
    np.testing.assert_allclose(by, ay, atol=1e-8)
    # Potential bis auf eine Konstante gleich -> gleiche Zeitverzögerungen
    d = lm.potential(X, Y, kw) - ours.potential(X, Y)
    assert np.ptp(d) < 1e-8


def test_nfw_matches_lenstronomy():
    n = NFW(0.3, 2.0)
    alpha_rs = float(n.deflection(2.0, 0.0)[0])
    lm = LensModel(["NFW"])
    ax, ay = lm.alpha(X, Y, [{"Rs": 2.0, "alpha_Rs": alpha_rs, "center_x": 0.0, "center_y": 0.0}])
    bx, by = n.deflection(X, Y)
    np.testing.assert_allclose(bx, ax, atol=1e-8)
    np.testing.assert_allclose(by, ay, atol=1e-8)
