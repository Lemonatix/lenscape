"""The six H0LiCOW lenses (Wong et al. 2020, MNRAS 498, 1420).

Redshifts, likelihood shapes and parameters follow exactly the
collaboration's public inference notebook
(github.com/shsuyu/H0LiCOW-public, H0_inference_code). The samples are
downloaded by ``scripts/fetch_h0licow.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .catalogs import DATA_DIR
from .distance_likelihoods import KDE1D, KDE2D, SkewLogNormal, scott_bandwidth

H0LICOW_DIR = DATA_DIR / "h0licow"


@dataclass
class TimeDelayLens:
    name: str
    z_d: float
    z_s: float
    reference: str
    # logL(D_d, D_dt) in Mpc; D_d is ignored when uses_dd is False
    loglike: Callable = field(repr=False)
    uses_dd: bool = False


def _read_dd_ddt(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", comment="#", names=("dd", "ddt"))


def load_h0licow(data_dir: Path | str | None = None) -> list[TimeDelayLens]:
    root = Path(data_dir) if data_dir else H0LICOW_DIR
    if not (root / "J1206_final.csv").exists():
        raise FileNotFoundError(f"H0LiCOW data missing in {root}. Run: python scripts/fetch_h0licow.py")
    lenses = []

    # B1608+656: analytic skewed log-normal fits for D_dt (Suyu+2010) and D_d (Jee+2019)
    b_ddt = SkewLogNormal(mu=7.0531390, sigma=0.2282395, lam=4000.0)
    b_dd = SkewLogNormal(mu=6.79671, sigma=0.1836, lam=334.2)
    lenses.append(TimeDelayLens("B1608+656", 0.6304, 1.394, "Suyu+2010, Jee+2019",
                                lambda dd, ddt: b_ddt.logpdf(ddt) + b_dd.logpdf(dd), uses_dd=True))

    # RXJ1131-1231: 2D KDE (Suyu+2014, Chen+2019)
    r = _read_dd_ddt(root / "RXJ1131_AO+HST_Dd_Ddt.dat")
    k_r = KDE2D(r["dd"], r["ddt"], bandwidth=20.0, nbins=80)
    lenses.append(TimeDelayLens("RXJ1131-1231", 0.295, 0.654, "Suyu+2014, Chen+2019", k_r.logpdf, uses_dd=True))

    # HE0435-1223: 1D KDE on D_dt (Wong+2017, Chen+2019)
    h = pd.read_csv(root / "HE0435_Ddt_AO+HST.dat", sep=r"\s+", skiprows=1, names=("ddt",))
    k_h = KDE1D(h["ddt"], bandwidth=20.0, nbins=400)
    lenses.append(TimeDelayLens("HE0435-1223", 0.4546, 1.693, "Wong+2017, Chen+2019",
                                lambda dd, ddt: k_h.logpdf(ddt)))

    # SDSS J1206+4332: 2D KDE (Birrer+2019); H0LiCOW uses scipy gaussian_kde -> Scott bandwidth
    j = pd.read_csv(root / "J1206_final.csv")
    bw = (scott_bandwidth(j["dd"], 2), scott_bandwidth(j["ddt"], 2))
    k_j = KDE2D(j["dd"], j["ddt"], bandwidth=bw, nbins=200)
    lenses.append(TimeDelayLens("J1206+4332", 0.745, 1.789, "Birrer+2019", k_j.logpdf, uses_dd=True))

    # WFI2033-4723: weighted D_dt samples, cut to 0 < D_dt < 8000 as in the notebook (Rusu+2020)
    w = pd.read_csv(root / "wfi2033_dt_bic.dat")
    w = w[(w["Dt"] > 0) & (w["Dt"] < 8000)]
    k_w = KDE1D(w["Dt"], weights=w["weight"], bandwidth=20.0, nbins=400)
    lenses.append(TimeDelayLens("WFI2033-4723", 0.6575, 1.662, "Rusu+2020",
                                lambda dd, ddt: k_w.logpdf(ddt)))

    # PG1115+080: 2D KDE (Chen+2019)
    p = _read_dd_ddt(root / "PG1115_AO+HST_Dd_Ddt.dat")
    k_p = KDE2D(p["dd"], p["ddt"], bandwidth=20.0, nbins=80)
    lenses.append(TimeDelayLens("PG1115+080", 0.311, 1.722, "Chen+2019", k_p.logpdf, uses_dd=True))

    return lenses
