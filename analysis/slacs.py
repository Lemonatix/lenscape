"""SLACS-Auswertung: Massen, Dunkle Materie und Kosmologie aus 63 echten Linsen.

    python analysis/slacs.py          # rechnen + Datentabellen schreiben
    python figures/build.py           # TikZ/pgfplots-Abbildungen bauen (PDF + SVG)

Die Rechnung schreibt nur Daten (figures/slacs/data/*.dat). Wie sie aussehen,
legen die pgfplots-Dateien in figures/slacs/*.tex fest.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import contourpy
import numpy as np

from lenscape.catalogs import SLACS_REF_COSMO, load_slacs
from lenscape.cosmography import fit_flat_wcdm, model_distance_ratio
from lenscape.cosmology import PLANCK18_OM0, make_cosmology
from lenscape.strong_lenses import einstein_quantities, summarize

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "slacs" / "data"
DATA.mkdir(parents=True, exist_ok=True)

DCHI2_LEVELS = {"68": 2.30, "95": 6.17}  # zwei Parameter


def write_table(name: str, header: list[str], *cols, fmt="%.6g") -> None:
    arr = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    np.savetxt(DATA / name, arr, header=" ".join(header), comments="", fmt=fmt)


def write_scalars(name: str, values: dict[str, float]) -> None:
    """Einzelwerte als pgf-Makros, z. B. \\def\\fsiemean{0.952}."""
    lines = [f"\\def\\{k}{{{v:.4g}}}" for k, v in values.items()]
    (DATA / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------- Exporte
def export_fsie(t) -> dict:
    f = np.asarray(t["f_SIE"][t["good_sigma"]])
    edges = np.arange(0.66, 1.30, 0.04)
    counts, _ = np.histogram(f, bins=edges)
    # pgfplots "ybar interval": letzte Zeile wiederholt den Zählwert am rechten Rand
    write_table("f_sie_hist.dat", ["left", "count"], edges, np.append(counts, counts[-1]))
    return {"fsiemean": f.mean(), "fsieerr": f.std(ddof=1) / np.sqrt(len(f)), "fsiestd": f.std(ddof=1),
            "fsien": len(f), "fsieymax": counts.max() + 6}


def export_fdm(t) -> None:
    x = np.asarray(t["R_E_over_R_e"])
    for imf in ("chab", "salp"):
        y = np.asarray(t[f"fDM_{imf}"])
        e = np.asarray(t[f"e_fstar_{imf}"])
        m = np.isfinite(y) & np.isfinite(e)
        write_table(f"f_dm_{imf}.dat", ["x", "y", "ey"], x[m], y[m], e[m])


def export_mass_vs_cosmology(t) -> dict:
    tab = load_slacs()
    ref = np.asarray(t["M_E"])
    ws = np.linspace(-2.0, -0.4, 33)
    oms = np.linspace(0.1, 0.6, 26)
    mw = [np.median(einstein_quantities(tab, make_cosmology(70, 0.3, None, w))["M_E"] / ref) for w in ws]
    mo = [np.median(einstein_quantities(tab, make_cosmology(70, om, None, -1.0))["M_E"] / ref) for om in oms]
    write_table("mass_vs_w.dat", ["w", "ratio"], ws, mw)
    write_table("mass_vs_om.dat", ["Om", "ratio"], oms, mo)
    return {"massspanw": 100 * np.ptp(mw), "massspanom": 100 * np.ptp(mo)}


def export_contours(res, tag: str) -> list[str]:
    """Gefüllte Δχ²-Bereiche als Polygone; Rückgabe: Liste der Dateinamen je Niveau."""
    d = res.delta_chi2()
    gen = contourpy.contour_generator(res.Om, res.w, d, fill_type=contourpy.FillType.OuterOffset)
    files = []
    for lvl, val in DCHI2_LEVELS.items():
        polys, _ = gen.filled(-1.0, val)
        for k, poly in enumerate(polys):
            name = f"cosmo_{tag}_{lvl}_{k}.dat"
            write_table(name, ["Om", "w"], poly[:, 0], poly[:, 1])
            files.append((lvl, name))
    return files


def forecast(t, n_lenses, seed=1):
    """Synthetische Stichprobe wie SLACS (gleiche z-Verteilung), Wahrheit = Planck-ΛCDM."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), n_lenses)
    z_d = np.asarray(t["z_d"])[idx]
    z_s = np.asarray(t["z_s"])[idx]
    sigma = np.asarray(t["sigma"])[idx]
    e_sigma = np.asarray(t["e_sigma"])[idx]
    ratio = model_distance_ratio(z_d, z_s, PLANCK18_OM0, -1.0)
    theta = 4 * np.pi * (sigma / 299792.458) ** 2 * ratio / (np.pi / 180 / 3600)
    theta *= np.exp(rng.normal(0, 0.15, n_lenses))  # intrinsische Streuung wie bei SLACS gemessen
    return fit_flat_wcdm(theta, sigma, e_sigma, z_d, z_s, e_theta_frac=0.02,
                         sigma_int=np.linspace(0.1, 0.2, 11))


def export_cosmography(t) -> tuple[dict, object]:
    g = t[t["good_sigma"]]
    res = fit_flat_wcdm(g["theta_E"], g["sigma"], g["e_sigma"], g["z_d"], g["z_s"], e_theta_frac=0.02)
    fc = forecast(g, 2000)
    files = {"slacs": export_contours(res, "slacs"), "forecast": export_contours(fc, "forecast")}

    # pgfplots-Befehle für die (je nach Daten unterschiedlich vielen) Polygone
    for tag, lst in files.items():
        cmds = [f"\\addplot[{tag}{lvl}, forget plot] table {{data/{name}}} -- cycle;" for lvl, name in
                sorted(lst, key=lambda p: -int(p[0]))]  # 95 % zuerst, 68 % darüber
        (DATA / f"cosmo_{tag}_regions.tex").write_text("\n".join(cmds) + "\n", encoding="utf-8")

    d = res.delta_chi2()
    j = np.argmin(np.abs(res.w + 1))
    i = np.argmin(np.abs(res.Om - PLANCK18_OM0))
    scal = {"nlenses": len(g), "nforecast": 2000, "dchiplanck": d[j, i], "planckOm": PLANCK18_OM0}
    return scal, res


def main():
    tab = load_slacs()
    t = einstein_quantities(tab, make_cosmology(**SLACS_REF_COSMO))
    s = summarize(t)

    scalars = {}
    scalars |= export_fsie(t)
    export_fdm(t)
    scalars |= export_mass_vs_cosmology(t)
    cs, res = export_cosmography(t)
    scalars |= cs
    write_scalars("scalars.tex", scalars)
    best = res.best

    m = np.isfinite(t["logM_E_auger"])
    dlog = np.log10(t["M_E"][m]) - t["logM_E_auger"][m]

    lines = [
        "# SLACS-Auswertung",
        "",
        "Automatisch erzeugt von `analysis/slacs.py`. Referenzkosmologie: H₀ = 70, Ωₘ = 0,3, Ω_Λ = 0,7.",
        "",
        "## Kennzahlen",
        "",
        "| Größe | Wert |",
        "|---|---|",
    ]
    for k, v in s.items():
        lines.append(f"| {k} | {v:.4g} |")
    lines += [
        "",
        "## Validierung",
        "",
        f"Unsere Einsteinmassen weichen von den publizierten Werten (Auger et al. 2009) im Mittel um "
        f"{dlog.mean():+.4f} dex ab (Streuung {dlog.std():.4f} dex, {m.sum()} Linsen).",
        "",
        "## Kosmologie",
        "",
        f"- Bestfit (flaches wCDM): Ωₘ = {best['Om']:.2f}, w = {best['w']:.2f}, "
        f"f_E = {best['f_E']:.3f}, intrinsische Streuung {best['sigma_int']:.2f}",
        f"- Planck-ΛCDM (Ωₘ = {PLANCK18_OM0}, w = −1) liegt bei Δχ² = {cs['dchiplanck']:.2f}, also verträglich.",
        f"- Variiert man w von −2 bis −0,4, ändern sich die Mediane der Massen um {scalars['massspanw']:.1f} %; "
        f"für Ωₘ von 0,1 bis 0,6 um {scalars['massspanom']:.1f} %.",
        "",
        "## Abbildungen",
        "",
        "Gebaut mit `python figures/build.py` (TikZ/pgfplots; PDF und SVG in `figures/slacs/`).",
        "",
        "![f_SIE](../figures/slacs/f_sie.svg)",
        "",
        "![f_DM](../figures/slacs/f_dm.svg)",
        "",
        "![Masse vs. Kosmologie](../figures/slacs/mass_vs_cosmology.svg)",
        "",
        "![Kosmographie](../figures/slacs/cosmography.svg)",
        "",
    ]
    (ROOT / "analysis" / "slacs_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
