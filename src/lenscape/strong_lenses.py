"""Physikalische Größen einer Stichprobe starker Linsen in beliebiger Kosmologie.

Aus Einsteinradius, Rotverschiebungen und Kosmologie folgt modellunabhängig die
projizierte Gesamtmasse innerhalb des Einsteinradius

    M_E = pi * Sigma_crit * (D_d theta_E)^2 .

Zusammen mit Licht und Sternmassen ergibt sich der Anteil Dunkler Materie im
Inneren der Galaxie:  f_DM(<R_E) = 1 - M_*(<R_E) / M_E .
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from .catalogs import SLACS_REF_COSMO
from .cosmology import LensGeometry, make_cosmology


def einstein_quantities(tab: Table, cosmo=None) -> Table:
    """Ergänzt Distanzen, M_E, sigma_SIE, f_SIE, M/L und f_DM.

    Erwartet die Spalten aus :func:`lenscape.catalogs.load_slacs`.
    Leuchtkräfte und Sternmassen sind in der SLACS-Referenzkosmologie angegeben
    und werden mit (D_L / D_L,ref)^2 auf ``cosmo`` umgerechnet.
    """
    cosmo = cosmo or make_cosmology()
    ref = make_cosmology(**SLACS_REF_COSMO)
    out = tab.copy()
    n = len(out)
    cols = {k: np.full(n, np.nan) for k in
            ["D_d", "D_s", "D_ds", "sigma_crit", "R_E_kpc", "M_E", "sigma_SIE", "lum_scale"]}
    for k, row in enumerate(out):
        g = LensGeometry(float(row["z_d"]), float(row["z_s"]), cosmo)
        th = float(row["theta_E"])
        cols["D_d"][k] = g.D_d.value
        cols["D_s"][k] = g.D_s.value
        cols["D_ds"][k] = g.D_ds.value
        cols["sigma_crit"][k] = g.sigma_crit.value
        cols["R_E_kpc"][k] = th * g.kpc_per_arcsec
        cols["M_E"][k] = g.mass_in_theta_E(th)
        cols["sigma_SIE"][k] = g.sigma_v_from_theta_E(th)
        dl = cosmo.luminosity_distance(row["z_d"]).value
        dl_ref = ref.luminosity_distance(row["z_d"]).value
        cols["lum_scale"][k] = (dl / dl_ref) ** 2
    for k, v in cols.items():
        out[k] = v

    out["f_SIE"] = out["sigma"] / out["sigma_SIE"]
    out["R_E_over_R_e"] = out["theta_E"] / out["r_e"]
    L_E = out["f_L_E"] * out["L_V"] * 1e9 * out["lum_scale"]
    out["ML_E"] = out["M_E"] / L_E  # M/L_V innerhalb theta_E in Sonneneinheiten

    if "logMstar_chab" in out.colnames:
        for imf in ("chab", "salp"):
            mstar = 10 ** out[f"logMstar_{imf}"] * out["lum_scale"]
            # Sternmasse innerhalb theta_E: Lichtanteil aus dem B-Spline-Modell (Bolton)
            out[f"Mstar_E_{imf}"] = mstar * out["f_L_E"]
            out[f"fDM_{imf}"] = 1 - out[f"Mstar_E_{imf}"] / out["M_E"]
    return out


def summarize(tab: Table) -> dict[str, float]:
    good = tab["good_sigma"] & np.isfinite(tab["sigma"])
    s = {
        "N": len(tab),
        "N_good_sigma": int(good.sum()),
        "median M_E [M_sun]": float(np.median(tab["M_E"])),
        "median R_E [kpc]": float(np.median(tab["R_E_kpc"])),
        "mean f_SIE (gute sigma)": float(np.mean(tab["f_SIE"][good])),
        "std f_SIE (gute sigma)": float(np.std(tab["f_SIE"][good], ddof=1)),
        "median M/L_V (<R_E)": float(np.nanmedian(tab["ML_E"])),
    }
    for imf in ("chab", "salp"):
        col = f"fDM_{imf}"
        if col in tab.colnames:
            s[f"median f_DM(<R_E) {imf}"] = float(np.nanmedian(tab[col]))
    return s
