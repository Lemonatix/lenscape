"""Einlesen echter Linsenkataloge (CDS/VizieR-Format).

Die Rohdateien lädt ``scripts/fetch_cds.py`` nach ``data/<katalog>/``. Jede
Tabelle wird über die mitgelieferte ReadMe (Byte-genaue Spaltenbeschreibung)
gelesen.

SLACS (Sloan Lens ACS Survey):
    Bolton et al. 2008, ApJ 682, 964  (J/ApJ/682/964)  -- SIE-Modelle, z, sigma
    Auger et al. 2009, ApJ 705, 1099  (J/ApJ/705/1099) -- Sternmassen
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy.io import ascii
from astropy.table import Table, join

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Kosmologie, mit der SLACS Leuchtkräfte und physikalische Radien angibt
SLACS_REF_COSMO = dict(H0=70.0, Om0=0.3, Ode0=0.7, w0=-1.0)


def read_cds(folder: Path | str, table: str) -> Table:
    folder = Path(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Einheiten wie '----' sind harmlos
        return ascii.read(folder / table, readme=folder / "ReadMe", format="cds")


def _filled(col, fill=np.nan):
    return np.ma.filled(np.ma.asarray(col, dtype=float), fill)


def load_slacs(data_dir: Path | str | None = None, with_auger: bool = True) -> Table:
    """Die 63 Grade-A-Linsen aus Bolton et al. (2008) mit SIE-Modell.

    Spalten (vereinheitlicht):
        name, z_d, z_s, sigma, e_sigma [km/s], theta_E [arcsec, SIE, Zwischenachse],
        q, pa [deg], r_e [arcsec], L_V [10^9 L_sun], f_L_E (Lichtanteil in theta_E),
        morph, ring, good_sigma
    und, falls vorhanden, aus Auger et al. (2009):
        logM_E_auger, R_E_kpc_auger, fstar_chab, e_fstar_chab, fstar_salp, e_fstar_salp,
        logMstar_chab, e_logMstar_chab, logMstar_salp, e_logMstar_salp
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    b = root / "slacs_bolton2008"
    t4 = read_cds(b, "table4.dat")
    t5 = read_cds(b, "table5.dat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # abweichende Spaltenbeschreibungen beim Join
        tab = join(t4, t5, keys="Name", join_type="inner")

    out = Table()
    out["name"] = [str(n) for n in tab["Name"]]
    out["z_d"] = _filled(tab["zFG"])
    out["z_s"] = _filled(tab["zBG"])
    out["sigma"] = _filled(tab["sigma"])
    out["e_sigma"] = _filled(tab["e_sigma"])
    out["theta_E"] = _filled(tab["bSIE"])
    out["q"] = _filled(tab["qSIE"])
    out["pa"] = _filled(tab["PASIE"])
    out["r_e"] = _filled(tab["Re"])
    out["L_V"] = _filled(tab["L(V555)"])
    out["f_L_E"] = _filled(tab["LSIE/LdV"])
    out["morph"] = [str(m) for m in tab["Mph"]]
    out["ring"] = np.array([str(v).strip() == "Yes" for v in tab["Ring?"]])
    out["good_sigma"] = np.array([str(v).strip() == "Yes" for v in tab["Good?"]])

    a = root / "slacs_auger2009"
    if with_auger and (a / "table4.dat").exists():
        t = read_cds(a, "table4.dat")
        aug = Table()
        aug["name"] = [str(n).strip() for n in t["SDSS"]]
        for src, dst in [("Mass", "logM_E_auger"), ("RE", "R_E_kpc_auger"),
                         ("Fc", "fstar_chab"), ("e_Fc", "e_fstar_chab"),
                         ("Fs", "fstar_salp"), ("e_Fs", "e_fstar_salp"),
                         ("logMc", "logMstar_chab"), ("e_logMc", "e_logMstar_chab"),
                         ("logMs", "logMstar_salp"), ("e_logMs", "e_logMstar_salp")]:
            aug[dst] = _filled(t[src])
        out = join(out, aug, keys="name", join_type="left")
        for col in out.colnames:
            if hasattr(out[col], "mask"):
                out[col] = out[col].filled(np.nan) if out[col].dtype.kind == "f" else out[col].filled()
    out.sort("name")
    return out
